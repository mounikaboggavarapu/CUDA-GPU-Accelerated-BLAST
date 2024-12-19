#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <cctype>
#include <iomanip>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Constants for scoring
const int MATCH_SCORE = 1;
const int GAP_COST = -1;
const int MISMATCH_COST = -1;


// Constants for k-mer filtering
const int KMER_LENGTH = 15;            // Length of k-mers
const int MIN_KMER_MATCHES = 4;       // Minimum number of matching k-mers to consider a reference
const int TOP_N = 5;                   // Number of top references to perform alignment on

const int N_MAX = 801;

// Global variables to accumulate timings
static double total_sw_gpu_time = 0.0; 
static double total_sw_time = 0.0;
static double total_kmer_search_gpu_time = 0.0;
static double total_kmer_search_time = 0.0; // Accumulate total time picking candidate sequences via k-mers
static double total_calculate_percent_identity_time = 0.0;
static double total_mem_transfer_time = 0.0;


// Struct for alignment result
struct AlignmentResult {
    int score;
    std::string aligned_seq1;
    std::string aligned_seq2;
    std::string accession_id; // Accessory ID of the reference
    int query_start;          // Start position in query
    int sbjct_start;          // Start position in subject
    double percent_identity;  // Percent identity of the alignment
};

// Struct to hold query results
struct QueryResult {
    std::string query_sequence;
    int num_candidate_references;               // Number of references meeting k-mer threshold
    std::vector<AlignmentResult> top_alignments; // Top N alignments
};



// Function declarations
std::vector<std::pair<std::string, std::string>> load_references(const std::string& filename);
std::vector<std::string> load_queries(const std::string& filename, int max_queries = -1);
std::vector<std::string> extract_kmers(const std::string& seq, int k);
bool is_valid_sequence(const std::string& seq);
void build_kmer_index(const std::vector<std::pair<std::string, std::string>>& references, int k, std::unordered_map<std::string, std::vector<int>>& kmer_index);
std::vector<AlignmentResult> smith_waterman_parallel(const std::string& seq1, const std::vector<std::string>& seq2_list, const std::vector<std::string>& seq2_accession_id_list);
double calculate_percent_identity(const std::string& seq1, const std::string& seq2);
std::string generate_alignment_indicator(const std::string& aligned_seq1, const std::string& aligned_seq2);
void print_alignment(const AlignmentResult& alignment);
void process_queries(const std::vector<std::string>& queries, 
                                 const std::unordered_map<std::string, std::vector<int>>& kmer_index, 
                                 const std::vector<std::pair<std::string, std::string>>& references, 
                                 std::vector<QueryResult>& results);

// Load references from file with sequence sanitization
std::vector<std::pair<std::string, std::string>> load_references(const std::string& filename) {
    std::vector<std::pair<std::string, std::string>> references;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open references file: " << filename << "\n";
        return references;
    }

    while (std::getline(file, line)) {
        if (!line.empty()) {
            std::istringstream iss(line);
            std::string accession_id, sequence;
            if (std::getline(iss, accession_id, ',') && std::getline(iss, sequence)) {
                // Sanitize sequence: replace invalid characters with 'N' and convert to uppercase
                std::transform(sequence.begin(), sequence.end(), sequence.begin(), [](char c) {
                    char upper = std::toupper(static_cast<unsigned char>(c));
                    if (upper == 'A' || upper == 'T' || upper == 'C' || upper == 'G' || upper == 'N') {
                        return upper;
                    } else {
                        return 'N';
                    }
                });
                references.emplace_back(accession_id, sequence);
            } else {
                std::cerr << "Warning: Invalid reference line format: " << line << "\n";
            }
        }
    }
    file.close();
    return references;
}


// Load queries from file with an optional limit on the number of queries
std::vector<std::string> load_queries(const std::string& filename, int max_queries) {
    std::vector<std::string> queries;
    std::ifstream file(filename);
    std::string line;
    int count = 0;

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open queries file: " << filename << "\n";
        return queries;
    }

    while (std::getline(file, line)) {
        if (!line.empty()) {
            // Validate sequence
            if (is_valid_sequence(line)) {
                // Sanitize sequence: replace invalid characters with 'N' and convert to uppercase
                std::transform(line.begin(), line.end(), line.begin(), [](char c) {
                    char upper = std::toupper(static_cast<unsigned char>(c));
                    if (upper == 'A' || upper == 'T' || upper == 'C' || upper == 'G' || upper == 'N') {
                        return upper;
                    } else {
                        return 'N';
                    }
                });
                queries.push_back(line);
                count++;
                if (max_queries > 0 && count >= max_queries) {
                    break;
                }
            } else {
                std::cerr << "Warning: Invalid characters in query sequence: " << line << "\n";
            }
        }
    }
    file.close();
    return queries;
}


// Extract k-mers from a sequence
std::vector<std::string> extract_kmers(const std::string& seq, int k) {
    std::vector<std::string> kmers;
    if (seq.size() < k) return kmers;
    kmers.reserve(seq.size() - k + 1); // Optimize memory allocation
    for (size_t i = 0; i <= seq.size() - k; ++i) {
        kmers.emplace_back(seq.substr(i, k));
    }
    return kmers;
}


// Validate that the sequence contains only valid DNA characters (A, T, C, G, N)
bool is_valid_sequence(const std::string& seq) {
    return std::all_of(seq.begin(), seq.end(), [](char c) {
        char upper = std::toupper(static_cast<unsigned char>(c));
        return upper == 'A' || upper == 'T' || upper == 'C' || upper == 'G' || upper == 'N';
    });
}

// Build k-mer index from references
void build_kmer_index(const std::vector<std::pair<std::string, std::string>>& references, int k, std::unordered_map<std::string, std::vector<int>>& kmer_index) {
    for (int i = 0; i < references.size(); ++i) {
        std::vector<std::string> kmers = extract_kmers(references[i].second, k);
        for (const auto& kmer : kmers) {
            kmer_index[kmer].push_back(i);
        }
    }
}


// Kernel Function
__global__ void smith_waterman_kernel(
    const char* seq1,
    const char* seq2,
    int m,
    int num_refs,
    int* score_matrices,
    int* max_scores,
    int* max_positions,
    char* aligned_seq1_array,
    char* aligned_seq2_array,
    int* alignment_lengths
)
{
    int n = N_MAX;

    int ref_string_idx = blockIdx.x;
    if(ref_string_idx >= num_refs) 
        return;

    __shared__ char s_ref_seq[N_MAX];

    int tid = threadIdx.x;

    const char* ref_seq = seq2 + ref_string_idx * N_MAX;
    int* score_matrix = score_matrices + ref_string_idx * (m+1) * (N_MAX + 1);
    int* max_score = max_scores + ref_string_idx;
    int* max_pos = max_positions + ref_string_idx * 2;

    // Pointers for alignment results
    char* aligned_seq1 = aligned_seq1_array + ref_string_idx * (m + N_MAX);  // Max possible alignment length
    char* aligned_seq2 = aligned_seq2_array + ref_string_idx * (m + N_MAX);
    int* alignment_length = alignment_lengths + ref_string_idx;

    // Load the reference sequence into shared memory
    for (int i = tid; i < N_MAX; i += blockDim.x) {
        s_ref_seq[i] = ref_seq[i];
    }
    __syncthreads();  // Ensure all threads have loaded the reference sequence

    for (int k = 2; k <= m + N_MAX; ++k) {
        int start_i = max(1, k - N_MAX);
        int end_i = min(m, k - 1);
        int num_elements = end_i - start_i + 1;

        for (int idx = tid; idx < num_elements; idx += blockDim.x) {
            int i = start_i + idx;
            int j = k - i;

            // Compute match/mismatch score
            int match = (seq1[i - 1] == ref_seq[j - 1]) ? MATCH_SCORE : MISMATCH_COST;

            // Calculate scores from neighboring cells
            int diag_score = score_matrix[(i - 1) * (N_MAX + 1) + (j - 1)] + match;
            int up_score = score_matrix[(i - 1) * (N_MAX + 1) + j] + GAP_COST;
            int left_score = score_matrix[i * (N_MAX + 1) + (j - 1)] + GAP_COST;

            int cell_score = max(0, max(diag_score, max(up_score, left_score)));

            score_matrix[i * (N_MAX + 1) + j] = cell_score;

             // Atomically update max score and position
            int old_max = atomicMax(max_score, cell_score);
            if (cell_score > old_max) {
                    atomicExch(&max_pos[0], i);
                    atomicExch(&max_pos[1], j);
                }
            }
            __syncthreads();
        }

        if(tid == 0)
        {
            int i = max_pos[0];
            int j = max_pos[1];
            int pos = 0;

            // Perform traceback until score is zero
            while (i > 0 && j > 0 && score_matrix[i * (n + 1) + j] > 0) {
                int current_score = score_matrix[i * (n + 1) + j];
                int diag_score = score_matrix[(i - 1) * (n + 1) + (j - 1)];
                int up_score = score_matrix[(i - 1) * (n + 1) + j];
                int left_score = score_matrix[i * (n + 1) + (j - 1)];

                if (current_score == diag_score + (seq1[i - 1] == s_ref_seq[j - 1] ? MATCH_SCORE : MISMATCH_COST)) {
                    aligned_seq1[pos] = seq1[i - 1];
                    aligned_seq2[pos] = s_ref_seq[j - 1];
                    --i;
                    --j;
                } else if (current_score == up_score + GAP_COST) {
                    aligned_seq1[pos] = seq1[i - 1];
                    aligned_seq2[pos] = '-';
                    --i;
                } else if (current_score == left_score + GAP_COST) {
                    aligned_seq1[pos] = '-';
                    aligned_seq2[pos] = s_ref_seq[j - 1];
                    --j;
                } else {
                    printf("Should not reach here in Smith-Waterman. Something is wrong");
                    break;
                }
                ++pos;
            }
            max_pos[0] = i+1;
            max_pos[1] = j+1;
            alignment_length[0] = pos;

        }
}

// Smith-Waterman alignment
std::vector<AlignmentResult> smith_waterman_parallel(const std::string& h_seq1, const std::vector<std::string>& h_seq2_list, const std::vector<std::string>& h_seq2_accession_id_list) {

    int m = (int)h_seq1.size();
    int num_refs = (int)h_seq2_list.size();

    auto start_time = std::chrono::high_resolution_clock::now();

    char* d_seq1;
    cudaMalloc(&d_seq1, m * sizeof(char));
    cudaMemcpy(d_seq1, h_seq1.data(), m * sizeof(char), cudaMemcpyHostToDevice);

    char* d_seq2;
    int total_ref_len = N_MAX * num_refs;

    cudaMalloc(&d_seq2, total_ref_len * sizeof(char));


    for(int i=0;i < num_refs;i++)
    {
        cudaMemcpy(d_seq2+i*N_MAX,h_seq2_list[i].data(),N_MAX*sizeof(char),cudaMemcpyHostToDevice); 
    }

    int* d_score_matrices;
    int score_matrix_size = (m + 1) * (N_MAX + 1);
    cudaMalloc(&d_score_matrices, num_refs * score_matrix_size * sizeof(int));

    int* d_max_scores;
    int* d_max_positions;
    cudaMalloc(&d_max_scores, num_refs * sizeof(int));
    cudaMalloc(&d_max_positions, num_refs * 2 * sizeof(int));


    // Initialize score matrices and max scores/positions
    cudaMemset(d_score_matrices, 0, num_refs * score_matrix_size * sizeof(int));
    cudaMemset(d_max_scores, 0, num_refs * sizeof(int));
    cudaMemset(d_max_positions, 0, num_refs * 2 * sizeof(int));

    //Device memory for alignment lengths
    char* d_aligned_seq1_arr;
    char* d_aligned_seq2_arr;
    int* d_alignment_lengths;
    int max_alignment_length = m + N_MAX;

    cudaMalloc(&d_aligned_seq1_arr, num_refs * max_alignment_length * sizeof(char));
    cudaMalloc(&d_aligned_seq2_arr, num_refs * max_alignment_length * sizeof(char));
    cudaMalloc(&d_alignment_lengths, num_refs * sizeof(int));

    cudaMemset(d_alignment_lengths, 0, num_refs * sizeof(int));

    auto end_time = std::chrono::high_resolution_clock::now();
    total_mem_transfer_time += std::chrono::duration<double>(end_time - start_time).count();

    int threadsPerBlock = N_MAX;
    int blocksPerGrid = num_refs;

    // Measure GPU kernel time using CUDA events
    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    cudaEventRecord(start_ev);

    smith_waterman_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_seq1,
        d_seq2,
        m,
        num_refs,
        d_score_matrices,
        d_max_scores,
        d_max_positions,
        d_aligned_seq1_arr,
        d_aligned_seq2_arr,
        d_alignment_lengths
    );

    cudaEventRecord(stop_ev);
    cudaEventSynchronize(stop_ev);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start_ev, stop_ev);
    total_sw_gpu_time += (gpu_ms / 1000.0);

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    // cudaDeviceSynchronize();

    std::vector<int> h_max_scores(num_refs);
    std::vector<int> h_alignment_lengths(num_refs);
    std::vector<AlignmentResult> results(num_refs);

    std::vector<int> h_max_positions(num_refs*2,0);


    // int *h_max_positions = (int*)malloc(num_refs * 2 * sizeof(int));
    // if (!h_max_positions) {
    //     fprintf(stderr, "Failed to allocate host memory for h_max_positions\n");
    //     return 1;
    // }
    start_time = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_max_scores.data(), d_max_scores, num_refs * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_alignment_lengths.data(), d_alignment_lengths, num_refs * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_max_positions, d_max_positions, num_refs * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max_positions.data(),d_max_positions,num_refs*2*sizeof(int),cudaMemcpyDeviceToHost);


    for (int i = 0; i < num_refs; ++i) {

        int alignment_length = h_alignment_lengths[i];
        char* h_aligned_seq1 = (char*)malloc(alignment_length * sizeof(char));
        char* h_aligned_seq2 = (char*)malloc(alignment_length * sizeof(char));
        // int* start_positions = h_max_positions + i*2;
        int start_query = h_max_positions[i * 2];
        int start_ref = h_max_positions[i * 2 + 1];
    
        cudaMemcpy(h_aligned_seq1, d_aligned_seq1_arr + i * max_alignment_length, alignment_length * sizeof(char), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_aligned_seq2, d_aligned_seq2_arr + i * max_alignment_length, alignment_length * sizeof(char), cudaMemcpyDeviceToHost);
    
        // Build the aligned sequences (reverse since they were constructed backwards)
        std::string aligned_seq1(h_aligned_seq1, alignment_length);
        std::string aligned_seq2(h_aligned_seq2, alignment_length);
        std::reverse(aligned_seq1.begin(), aligned_seq1.end());
        std::reverse(aligned_seq2.begin(), aligned_seq2.end());


        auto start_time = std::chrono::high_resolution_clock::now();
        double percent_identity = calculate_percent_identity(h_aligned_seq1,h_aligned_seq2);
        auto end_time = std::chrono::high_resolution_clock::now();
        total_calculate_percent_identity_time += std::chrono::duration<double>(end_time - start_time).count();
    
        // Store the result
        results[i] = {h_max_scores[i], aligned_seq1, aligned_seq2,h_seq2_accession_id_list[i],start_query,start_ref,percent_identity};
    
        free(h_aligned_seq1);
        free(h_aligned_seq2);
    }

    cudaFree(d_seq1);
    cudaFree(d_seq2);
    cudaFree(d_score_matrices);
    cudaFree(d_max_scores);
    cudaFree(d_max_positions);
    cudaFree(d_aligned_seq1_arr);
    cudaFree(d_aligned_seq2_arr);
    cudaFree(d_alignment_lengths);
    end_time = std::chrono::high_resolution_clock::now();
    total_mem_transfer_time += std::chrono::duration<double>(end_time - start_time).count();
    // free(h_max_positions);
    // free(h_alignment_lengths);
    // free(h_max_scores);

    return results;
}

// Calculate percent identity based on alignment
double calculate_percent_identity(const std::string& seq1, const std::string& seq2) {
    int matches = 0;
    int aligned_length = 0;

    for (size_t i = 0; i < seq1.length() && i < seq2.length(); ++i) {
        if (seq1[i] == '-' || seq2[i] == '-') {
            aligned_length++;
        } else {
            aligned_length++;
            if (seq1[i] == seq2[i]) {
                matches++;
            }
        }
    }
    
    return (aligned_length > 0) ? (static_cast<double>(matches) / aligned_length) * 100.0 : 0.0;
}



// Function to generate alignment indicator line
std::string generate_alignment_indicator(const std::string& aligned_seq1, const std::string& aligned_seq2) {
    std::string indicator = "";
    size_t length = std::min(aligned_seq1.length(), aligned_seq2.length());
    for (size_t i = 0; i < length; ++i) {
        if (aligned_seq1[i] == aligned_seq2[i] && aligned_seq1[i] != '-') {
            indicator += "|";
        }
        else {
            indicator += " ";
        }
    }
    return indicator;
}


// Function to format and print alignment with positions
void print_alignment(const AlignmentResult& alignment) {
    if (alignment.score == 0 || alignment.aligned_seq1.empty() || alignment.aligned_seq2.empty()) {
        std::cout << "No alignment available.\n\n";
        return;
    }
 
    // Recalculate matches and alignment length
    int matches = 0;
    int alignment_length = 0;
    for (size_t i = 0; i < alignment.aligned_seq1.size() && i < alignment.aligned_seq2.size(); ++i) {
        char q_char = alignment.aligned_seq1[i];
        char s_char = alignment.aligned_seq2[i];
        if (q_char != '-' && s_char != '-') {
            alignment_length++;
            if (q_char == s_char) {
                matches++;
            }
        }
    }
 
    double percent = (alignment_length > 0) ? (100.0 * matches / alignment_length) : 0.0;
    // Round percentage to nearest integer
    int rounded_percent = static_cast<int>(std::round(percent));
 
    // Print the calculated percent identity in the desired format
    std::cout << "Percent identity = " << matches << "/" << alignment_length
              << " (" << rounded_percent << "%)\n";
 
    // Calculate aligned positions
    int query_start = alignment.query_start;
    int sbjct_start = alignment.sbjct_start;
 
    // Print the alignment in blocks of 60 characters
    const int BLOCK_SIZE = 60;
    size_t alignment_length_total = alignment.aligned_seq1.length();
    size_t num_blocks = (alignment_length_total + BLOCK_SIZE - 1) / BLOCK_SIZE;
 
    for (size_t block = 0; block < num_blocks; ++block) {
        size_t start_idx = block * BLOCK_SIZE;
        size_t end_idx = std::min(start_idx + BLOCK_SIZE, alignment_length_total);
 
        // Extract the current block
        std::string query_block = alignment.aligned_seq1.substr(start_idx, end_idx - start_idx);
        std::string sbjct_block = alignment.aligned_seq2.substr(start_idx, end_idx - start_idx);
        std::string indicator_block = generate_alignment_indicator(query_block, sbjct_block);
 
        // Calculate positions
        int q_block_start = query_start;
        int s_block_start = sbjct_start;
 
        // Adjust start positions for gaps in the previous blocks
        for (size_t i = 0; i < start_idx; ++i) {
            if (alignment.aligned_seq1[i] != '-') q_block_start++;
            if (alignment.aligned_seq2[i] != '-') s_block_start++;
        }
 
        int q_block_end = q_block_start;
        int s_block_end = s_block_start;
 
        for (char c : query_block) {
            if (c != '-') q_block_end++;
        }
        q_block_end = (q_block_end == q_block_start) ? q_block_start : q_block_end - 1;
 
        for (char c : sbjct_block) {
            if (c != '-') s_block_end++;
        }
        s_block_end = (s_block_end == s_block_start) ? s_block_start : s_block_end - 1;
 
        // Generate Query line
        std::ostringstream query_line;
        query_line << "Query  " << q_block_start << "  " << query_block << "  " << q_block_end;
        std::string query_line_str = query_line.str();
        std::cout << query_line_str << "\n";
 
        // Calculate padding for Indicator line
        std::ostringstream prefix_stream;
        prefix_stream << "Query  " << q_block_start << "  ";
        std::string prefix = prefix_stream.str();
        size_t prefix_length = prefix.length();
 
        // Generate Indicator line with matching padding
        std::string indicator_line(prefix_length, ' ');
        indicator_line += indicator_block;
        std::cout << indicator_line << "\n";
 
        // Generate Sbjct line
        std::ostringstream sbjct_line;
        sbjct_line << "Sbjct  " << s_block_start << "  " << sbjct_block << "  " << s_block_end;
        std::string sbjct_line_str = sbjct_line.str();
        std::cout << sbjct_line_str << "\n\n";
    }
}

// A simple hash function for k-mers
// This maps a fixed-length k-mer to a hash value.
unsigned int hash_kmer(const char *kmer, int k, unsigned int table_size) {
    // Polynomial rolling hash
    unsigned long long h = 0;
    unsigned long long base = 131ULL;
    for (int i = 0; i < k; i++) {
        unsigned char c = (unsigned char)kmer[i];
        h = h * base + c;
    }
    return (unsigned int)(h % table_size);
}

// Find a suitable table size (for simplicity, pick a size ~2 * num_kmers, ensure it's prime-ish or just large enough)
static unsigned int pick_table_size(unsigned int num_kmers) {
    // For simplicity, just pick the next power of two or a larger number.
    // A prime might be better, but let's pick something large:
    unsigned int size = 1;
    while (size < num_kmers * 2) {
        size <<= 1; 
    }
    // You may pick a prime here, but power of two works with good hash distributions.
    return size;
}

void build_gpu_friendly_hash_table(
    const std::unordered_map<std::string, std::vector<int>>& kmer_index,
    std::vector<char>& h_hash_kmers,
    std::vector<int>& h_hash_ref_starts,
    std::vector<int>& h_hash_ref_counts,
    std::vector<int>& h_ref_indices,
    unsigned int& table_size
) {
    int num_kmers = (int)kmer_index.size();
    // Count total references
    size_t total_refs = 0;
    for (auto &kv : kmer_index) {
        total_refs += kv.second.size();
    }

    // Compute table size
    table_size = pick_table_size(num_kmers);

    // Prepare arrays
    h_hash_kmers.resize(table_size * KMER_LENGTH, 0);
    h_hash_ref_starts.resize(table_size, -1);
    h_hash_ref_counts.resize(table_size, -1);
    h_ref_indices.resize(total_refs);

    // Fill references
    // We'll build a consecutive list of reference indices and remember their starts and counts
    size_t ref_pos = 0;
    for (auto &kv : kmer_index) {
        const std::string &kmer = kv.first;
        const auto &refs = kv.second;

        // Insert into hash table via linear probing
        unsigned int hval = hash_kmer(kmer.c_str(), KMER_LENGTH, table_size);
        unsigned int orig = hval;
        bool inserted = false;
        do {
            if (h_hash_ref_counts[hval] == -1) {
                // Empty slot
                // Copy kmer
                for (int i = 0; i < KMER_LENGTH; i++) {
                    h_hash_kmers[hval * KMER_LENGTH + i] = kmer[i];
                }

                // Store start and count
                h_hash_ref_starts[hval] = (int)ref_pos;
                h_hash_ref_counts[hval] = (int)refs.size();

                // Copy references
                for (size_t r = 0; r < refs.size(); r++) {
                    h_ref_indices[ref_pos + r] = refs[r];
                }
                ref_pos += refs.size();
                inserted = true;
                break;
            } else {
                // Collision
                hval = (hval + 1) % table_size;
            }
        } while (hval != orig);

        if (!inserted) {
            std::cerr << "Error: Hash table full or too many collisions. Increase table size.\n";
            exit(1);
        }
    }
}

__device__ unsigned int d_hash_kmer(const char *kmer, int k, unsigned int table_size) {
    unsigned long long h = 0ULL;
    unsigned long long base = 131ULL;
    for (int i = 0; i < k; i++) {
        unsigned char c = (unsigned char)kmer[i];
        h = h * base + c;
    }
    return (unsigned int)(h % table_size);
}

__global__ void parallel_kmer_search_kernel(
    const char* d_hash_kmers,   // [table_size * KMER_LENGTH]
    const int* d_hash_ref_starts,
    const int* d_hash_ref_counts,
    const int* d_ref_indices,
    unsigned int table_size,
    const char* d_query_kmers,
    const int* d_query_kmer_starts,
    const int* d_query_kmer_counts,
    int num_queries,
    int* d_reference_scores,
    int num_refs,
    const int* d_tid_to_q_id,
    const int* d_tid_to_q_kmer_idx,
    int total_query_kmers
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_query_kmers) return;

    int q_id = d_tid_to_q_id[tid];
    int q_kmer_idx = d_tid_to_q_kmer_idx[tid];

    int q_kmer_start = d_query_kmer_starts[q_id] + q_kmer_idx * KMER_LENGTH;

    char q_kmer[KMER_LENGTH];
    for (int i = 0; i < KMER_LENGTH; i++) {
        q_kmer[i] = d_query_kmers[q_kmer_start + i];
    }

    // Hash and lookup
    unsigned int hval = d_hash_kmer(q_kmer, KMER_LENGTH, table_size);
    unsigned int orig = hval;
    while (true) {
        int count = d_hash_ref_counts[hval];
        if (count == -1) {
            // Empty slot found, no match
            break;
        } else {
            // Check if this slot matches the k-mer
            bool match = true;
            int base = hval * KMER_LENGTH;
            #pragma unroll
            for (int c = 0; c < KMER_LENGTH; c++) {
                if (d_hash_kmers[base + c] != q_kmer[c]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                // We found our k-mer
                int start = d_hash_ref_starts[hval];
                for (int r = 0; r < count; r++) {
                    int ref_id = d_ref_indices[start + r];
                    atomicAdd(&d_reference_scores[q_id * num_refs + ref_id], 1);
                }
                break; // No need to continue probing
            } else {
                // Collision, keep probing
                hval = (hval + 1) % table_size;
                if (hval == orig) {
                    // Came full circle, no match
                    break;
                }
            }
        }
    }
}

void Parallel_k_mer_search(
    const std::vector<std::string>& queries,
    const std::unordered_map<std::string, std::vector<int>>& kmer_index,
    int num_refs,
    std::vector<std::unordered_map<int,int>>& per_query_reference_scores
) {
    // Build a GPU-friendly hash table
    std::vector<char> h_hash_kmers;
    std::vector<int> h_hash_ref_starts, h_hash_ref_counts, h_ref_indices;
    unsigned int table_size;
    build_gpu_friendly_hash_table(kmer_index, h_hash_kmers, h_hash_ref_starts, h_hash_ref_counts, h_ref_indices, table_size);

    // Flatten query k-mers
    std::vector<int> h_query_kmer_starts(queries.size()), h_query_kmer_counts(queries.size());
    int total_query_kmers = 0;
    for (size_t i = 0; i < queries.size(); i++) {
        std::vector<std::string> qkmers = extract_kmers(queries[i], KMER_LENGTH);
        h_query_kmer_starts[i] = total_query_kmers * KMER_LENGTH;
        h_query_kmer_counts[i] = (int)qkmers.size();
        total_query_kmers += (int)qkmers.size();
    }

    std::vector<char> h_query_kmers(total_query_kmers * KMER_LENGTH);
    {
        int pos = 0;
        for (size_t i = 0; i < queries.size(); i++) {
            std::vector<std::string> qkmers = extract_kmers(queries[i], KMER_LENGTH);
            for (const auto &km : qkmers) {
                for (int c = 0; c < KMER_LENGTH; c++) {
                    h_query_kmers[pos * KMER_LENGTH + c] = km[c];
                }
                pos++;
            }
        }
    }

    // Map tid -> (q_id, q_kmer_idx)
    std::vector<int> h_tid_to_q_id(total_query_kmers), h_tid_to_q_kmer_idx(total_query_kmers);
    {
        int pos = 0;
        for (int q = 0; q < (int)queries.size(); q++) {
            int count = h_query_kmer_counts[q];
            for (int k_idx = 0; k_idx < count; k_idx++) {
                h_tid_to_q_id[pos] = q;
                h_tid_to_q_kmer_idx[pos] = k_idx;
                pos++;
            }
        }
    }

    // Allocate device memory
    char* d_hash_kmers;
    int *d_hash_ref_starts, *d_hash_ref_counts, *d_ref_indices;
    char *d_query_kmers;
    int *d_query_kmer_starts, *d_query_kmer_counts;
    int *d_reference_scores;
    int *d_tid_to_q_id, *d_tid_to_q_kmer_idx;

    int num_refs_total = num_refs;
    size_t total_refs = h_ref_indices.size();

    cudaMalloc((void**)&d_hash_kmers, h_hash_kmers.size()*sizeof(char));
    cudaMalloc((void**)&d_hash_ref_starts, table_size*sizeof(int));
    cudaMalloc((void**)&d_hash_ref_counts, table_size*sizeof(int));
    cudaMalloc((void**)&d_ref_indices, total_refs*sizeof(int));

    cudaMalloc((void**)&d_query_kmers, h_query_kmers.size()*sizeof(char));
    cudaMalloc((void**)&d_query_kmer_starts, queries.size()*sizeof(int));
    cudaMalloc((void**)&d_query_kmer_counts, queries.size()*sizeof(int));

    cudaMalloc((void**)&d_tid_to_q_id, total_query_kmers*sizeof(int));
    cudaMalloc((void**)&d_tid_to_q_kmer_idx, total_query_kmers*sizeof(int));

    cudaMalloc((void**)&d_reference_scores, queries.size()*num_refs_total*sizeof(int));
    cudaMemset(d_reference_scores, 0, queries.size()*num_refs_total*sizeof(int));

    // Copy data
    cudaMemcpy(d_hash_kmers, h_hash_kmers.data(), h_hash_kmers.size()*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_ref_starts, h_hash_ref_starts.data(), table_size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash_ref_counts, h_hash_ref_counts.data(), table_size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_indices, h_ref_indices.data(), total_refs*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_query_kmers, h_query_kmers.data(), h_query_kmers.size()*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_kmer_starts, h_query_kmer_starts.data(), queries.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_kmer_counts, h_query_kmer_counts.data(), queries.size()*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_tid_to_q_id, h_tid_to_q_id.data(), total_query_kmers*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tid_to_q_kmer_idx, h_tid_to_q_kmer_idx.data(), total_query_kmers*sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (total_query_kmers + blockSize - 1) / blockSize;

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    cudaEventRecord(start_ev);

    parallel_kmer_search_kernel<<<gridSize, blockSize>>>(
        d_hash_kmers,
        d_hash_ref_starts,
        d_hash_ref_counts,
        d_ref_indices,
        table_size,
        d_query_kmers,
        d_query_kmer_starts,
        d_query_kmer_counts,
        (int)queries.size(),
        d_reference_scores,
        num_refs_total,
        d_tid_to_q_id,
        d_tid_to_q_kmer_idx,
        total_query_kmers
    );

    cudaEventRecord(stop_ev);
    cudaEventSynchronize(stop_ev);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start_ev, stop_ev);
    total_kmer_search_gpu_time += (gpu_ms / 1000.0);

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);

    // Copy results back
    std::vector<int> h_reference_scores(queries.size()*num_refs_total, 0);
    cudaMemcpy(h_reference_scores.data(), d_reference_scores, queries.size()*num_refs_total*sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_hash_kmers);
    cudaFree(d_hash_ref_starts);
    cudaFree(d_hash_ref_counts);
    cudaFree(d_ref_indices);
    cudaFree(d_query_kmers);
    cudaFree(d_query_kmer_starts);
    cudaFree(d_query_kmer_counts);
    cudaFree(d_tid_to_q_id);
    cudaFree(d_tid_to_q_kmer_idx);
    cudaFree(d_reference_scores);

    // Populate results
    for (size_t q = 0; q < queries.size(); q++) {
        std::unordered_map<int,int> score_map;
        for (int r = 0; r < num_refs_total; r++) {
            int score = h_reference_scores[q*num_refs_total + r];
            if (score > 0) {
                score_map[r] = score;
            }
        }
        per_query_reference_scores[q] = std::move(score_map);
    }
}

// Process queries in parallel using multiple threads
void process_queries(const std::vector<std::string>& queries, 
    const std::unordered_map<std::string, std::vector<int>>& kmer_index, 
    const std::vector<std::pair<std::string, std::string>>& references, 
    std::vector<QueryResult>& results) {

    size_t num_queries = queries.size();
    results.resize(num_queries);

    // Start K-mer Search Selection (GPU)
    auto kmer_search_start = std::chrono::high_resolution_clock::now();

    // Parallel K-mer search
    std::vector<std::unordered_map<int,int>> per_query_reference_scores(num_queries);
    Parallel_k_mer_search(queries, kmer_index, (int)references.size(), per_query_reference_scores);

    auto kmer_search_end = std::chrono::high_resolution_clock::now();
    total_kmer_search_time += std::chrono::duration<double>(kmer_search_end - kmer_search_start).count();

    for (size_t i = 0; i < num_queries; ++i) {
        const std::string& query = queries[i];
        const auto& reference_scores = per_query_reference_scores[i];


        // Initialize QueryResult
        QueryResult qr;
        qr.query_sequence = query;
        qr.num_candidate_references = 0;
        qr.top_alignments = {};

        if (reference_scores.empty()) {
            // No matching references found
            results[i] = qr;
            continue;
        }

        // Count total candidate references meeting the k-mer threshold
        int total_candidates = 0;
        for (const auto& score_pair : reference_scores) {
            int count = score_pair.second;
            if (count >= MIN_KMER_MATCHES) {
                total_candidates++;
            }
        }

        qr.num_candidate_references = total_candidates;

        if (total_candidates == 0) {
            // No references meet the minimum k-mer match threshold
            results[i] = qr;
            continue;
        }

        // Filter references based on MIN_KMER_MATCHES
        std::vector<std::pair<int, int>> sorted_refs;
        for (const auto& score_pair : reference_scores) {
            int ref_idx = score_pair.first;
            int count = score_pair.second;
            if (count >= MIN_KMER_MATCHES) {
                sorted_refs.emplace_back(ref_idx, count);
            }
        }

        // Sort the filtered references by number of matching k-mers in descending order
        std::sort(sorted_refs.begin(), sorted_refs.end(),
        [&](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.second > b.second;
        });

        // Limit to TOP_N references
        if (sorted_refs.size() > TOP_N) {
            sorted_refs.resize(TOP_N);
        }


        std::vector<std::string> candidate_refs;
        std::vector<std::string> candidate_accession_ids;
        candidate_refs.reserve(sorted_refs.size());
        candidate_accession_ids.reserve(sorted_refs.size());


        // Perform Smith-Waterman alignment with each candidate and store the alignments
        for (const auto& score_pair : sorted_refs) {
            int ref_idx = score_pair.first;
            const auto& reference = references[ref_idx];
            candidate_refs.push_back(reference.second);
            candidate_accession_ids.push_back(reference.first);
            
        }
        auto sw_start = std::chrono::high_resolution_clock::now();
        std::vector<AlignmentResult> sw_results = smith_waterman_parallel(query, candidate_refs, candidate_accession_ids);
        auto sw_end = std::chrono::high_resolution_clock::now();

        total_sw_time += std::chrono::duration<double>(sw_end - sw_start).count();

        // Only consider alignments with percent identity >= 80%
        for(const auto& alignment : sw_results)
        {
            if (alignment.percent_identity >= 80.0) {
                qr.top_alignments.push_back(alignment);
            }
        }

        // Store the result
        results[i] = qr;
    }
}

int main() {
    try {
        // File paths (adjust these paths as necessary)
        std::string references_file = "input/lastest_proj3_db.txt";
        std::string queries_file = "input/800_proj3query_latest.txt";

        // Load references and queries
        std::vector<std::pair<std::string, std::string>> references = load_references(references_file);
        std::vector<std::string> queries = load_queries(queries_file); // Load all queries


        if (references.empty()) {
            std::cerr << "Error: No references loaded. Please check your references file.\n";
            return 1;
        }

        if (queries.empty()) {
            std::cerr << "Error: No queries loaded. Please check your queries file.\n";
            return 1;
        }

        // Time building the k-mer index
        auto kmer_start_time = std::chrono::high_resolution_clock::now();
        // Build k-mer index
        std::unordered_map<std::string, std::vector<int>> kmer_index;
        build_kmer_index(references, KMER_LENGTH, kmer_index);
        auto kmer_end_time = std::chrono::high_resolution_clock::now();
        double kmer_build_time = std::chrono::duration<double>(kmer_end_time - kmer_start_time).count();


        // Prepare a vector to hold results
        std::vector<QueryResult> results;

        // Start timing (only for processing)
        auto start_time = std::chrono::high_resolution_clock::now();

        // Process queries in parallel
        process_queries(queries, kmer_index, references, results);

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();

        // Calculate duration and throughput
        std::chrono::duration<double> duration = end_time - start_time;
        double throughput = queries.size() / duration.count();

        // Calculate total candidate references for average
        long long total_candidate_references = 0;
        for (const auto& qr : results) {
            total_candidate_references += qr.num_candidate_references;
        }
        double average_candidate_references = queries.empty() ? 0.0 : static_cast<double>(total_candidate_references) / queries.size();

        // Print results in query order (printing excluded from timing)
        for (size_t idx = 0; idx < results.size(); ++idx) {
            const auto& qr = results[idx];
            std::cout << "----------------------------------------\n";
            std::cout << "Query " << (idx + 1) << ":\n";
            std::cout << "Query sequence: " << qr.query_sequence << "\n";
            std::cout << "Number of candidate references after k-mer filtering: " << qr.num_candidate_references << "\n";

            if (!qr.top_alignments.empty()) {
                for (size_t aln_idx = 0; aln_idx < qr.top_alignments.size(); ++aln_idx) {
                    const auto& alignment = qr.top_alignments[aln_idx];
                    std::cout << "Top " << (aln_idx + 1) << " Reference sequence (accession ID = " << alignment.accession_id << "):\n";
                    std::cout << "Smith-Waterman Score = " << alignment.score << "\n";
                    // std::cout << "Percent identity = " << static_cast<int>(std::round(alignment.percent_identity)) << "%\n";
                    std::cout << "Alignment:\n";

                    // Print alignment
                    print_alignment(alignment);
                }
            }
            else {
                std::cout << "No alignments with percent identity >= 80% found among top references.\n\n";
            }
        }

        std::cout << "----------------------------------------\n";

        std::cout << "Number of queries loaded: " << queries.size() << "\n";

        // Print average candidate references
        std::cout << "Average number of candidate references across all queries: " 
                  << std::fixed << std::setprecision(2) << average_candidate_references << "\n\n";

        // Print performance metrics
        std::cout << "Total Execution Time (Processing Only): " << std::fixed << std::setprecision(3) << total_sw_gpu_time + total_kmer_search_gpu_time + total_calculate_percent_identity_time << " seconds\n";
        std::cout << "Throughput: " << std::fixed << std::setprecision(3) << queries.size()/(total_sw_gpu_time + total_kmer_search_gpu_time + total_calculate_percent_identity_time) << " queries/second\n";
        // std::cout << "Total SW kernel time: " << total_sw_gpu_time << " s\n";
        // // std::cout << "Total SW time" << total_sw_time << "s\n";
        // std::cout << "Total time building K-mer Index: " << kmer_build_time << " s\n";
        // std::cout << "Total K-mer search kernel time: " << total_kmer_search_gpu_time << "s\n";
        // std::cout << "Total time picking candidate references via k-mers: " << total_kmer_search_time << " s\n";
        // std::cout << "Total time calculating percent identity: " << total_calculate_percent_identity_time << " s\n";
        // std::cout << "Total time allocating, copying, freeing memory for CUDA kernel : " << total_mem_transfer_time - total_calculate_percent_identity_time << " s\n";
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << "\n";
        return 1;
    }
}
