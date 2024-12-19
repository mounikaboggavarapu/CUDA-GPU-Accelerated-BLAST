#include <iostream>
#include <cmath>
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

// Constants for scoring
const int MATCH_SCORE = 1;
const int GAP_COST = -1;
const int MISMATCH_COST = -1;

// Constants for k-mer filtering
const int KMER_LENGTH = 15;            // Length of k-mers
const int MIN_KMER_MATCHES = 4;       // Minimum number of matching k-mers to consider a reference
const int TOP_N = 5;                   // Number of top references to perform alignment on

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
AlignmentResult smith_waterman(const std::string& seq1, const std::string& seq2, const std::string& accession_id);
double calculate_percent_identity(const std::string& seq1, const std::string& seq2);
std::string generate_alignment_indicator(const std::string& aligned_seq1, const std::string& aligned_seq2);
void print_alignment(const AlignmentResult& alignment);
void process_queries_in_parallel(const std::vector<std::string>& queries,
                                 const std::unordered_map<std::string, std::vector<int>>& kmer_index,
                                 const std::vector<std::pair<std::string, std::string>>& references,
                                 std::vector<QueryResult>& results);

// Function implementations

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

// Smith-Waterman alignment algorithm
AlignmentResult smith_waterman(const std::string& seq1, const std::string& seq2, const std::string& accession_id) {
    int m = seq1.size();
    int n = seq2.size();
    std::vector<std::vector<int>> score_matrix(m + 1, std::vector<int>(n + 1, 0));
    int max_score = 0, max_i = -1, max_j = -1;

    // Fill the score matrix
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            int match = score_matrix[i - 1][j - 1] + ((seq1[i - 1] == seq2[j - 1] && seq1[i - 1] != 'N' && seq2[j - 1] != 'N') ? MATCH_SCORE : MISMATCH_COST);
            int delete_ = score_matrix[i - 1][j] + GAP_COST;
            int insert = score_matrix[i][j - 1] + GAP_COST;
            score_matrix[i][j] = std::max({0, match, delete_, insert});

            if (score_matrix[i][j] > max_score) {
                max_score = score_matrix[i][j];
                max_i = i;
                max_j = j;
            }
        }
    }

    // Traceback
    std::string aligned_seq1 = "", aligned_seq2 = "";
    int i = max_i, j = max_j;
    int query_start = i, sbjct_start = j;

    while (i > 0 && j > 0 && score_matrix[i][j] > 0) {
        if (score_matrix[i][j] == score_matrix[i - 1][j - 1] + ((seq1[i - 1] == seq2[j - 1] && seq1[i - 1] != 'N' && seq2[j - 1] != 'N') ? MATCH_SCORE : MISMATCH_COST)) {
            aligned_seq1 = seq1[i - 1] + aligned_seq1;
            aligned_seq2 = seq2[j - 1] + aligned_seq2;
            --i;
            --j;
        }
        else if (score_matrix[i][j] == score_matrix[i - 1][j] + GAP_COST) {
            aligned_seq1 = seq1[i - 1] + aligned_seq1;
            aligned_seq2 = "-" + aligned_seq2;
            --i;
        }
        else {
            aligned_seq1 = "-" + aligned_seq1;
            aligned_seq2 = seq2[j - 1] + aligned_seq2;
            --j;
        }
    }

    // Add remaining unaligned parts
    while (i > 0) {
        aligned_seq1 = seq1[i - 1] + aligned_seq1;
        aligned_seq2 = "-" + aligned_seq2;
        --i;
    }
    while (j > 0) {
        aligned_seq1 = "-" + aligned_seq1;
        aligned_seq2 = seq2[j - 1] + aligned_seq2;
        --j;
    }

    // Update start positions
    // After traceback, i and j are at the positions before the start of alignment
    query_start = i + 1;
    sbjct_start = j + 1;

    // Calculate percent identity
    double percent_identity = 0.0;
    int matches = 0;
    int alignment_length = 0;
    for (size_t pos = 0; pos < aligned_seq1.length() && pos < aligned_seq2.length(); ++pos) {
        if (aligned_seq1[pos] == aligned_seq2[pos] && aligned_seq1[pos] != '-') {
            matches++;
        }
        if (aligned_seq1[pos] != '-' && aligned_seq2[pos] != '-') {
            alignment_length++;
        }
    }
    if (alignment_length > 0) {
        percent_identity = (static_cast<double>(matches) / alignment_length) * 100.0;
    }

    return {max_score, aligned_seq1, aligned_seq2, accession_id, query_start, sbjct_start, percent_identity};
}

// Calculate percent identity based on alignment
double calculate_percent_identity(const std::string& seq1, const std::string& seq2) {
    int matches = 0;
    int alignment_length = 0;

    // Count matches over the aligned sequences
    for (size_t i = 0; i < seq1.length() && i < seq2.length(); ++i) {
        if (seq1[i] == seq2[i] && seq1[i] != '-') {
            ++matches;
        }
        if (seq1[i] != '-' && seq2[i] != '-') {
            ++alignment_length;
        }
    }

    if (alignment_length == 0) return 0.0;

    return (static_cast<double>(matches) / alignment_length) * 100.0;
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


// Process queries in parallel using multiple threads
void process_queries_in_parallel(const std::vector<std::string>& queries,
                                 const std::unordered_map<std::string, std::vector<int>>& kmer_index,
                                 const std::vector<std::pair<std::string, std::string>>& references,
                                 std::vector<QueryResult>& results) {
    size_t num_queries = queries.size();
    unsigned int hardware_threads = std::thread::hardware_concurrency();
    unsigned int num_threads = hardware_threads > 0 ? hardware_threads : 4;
    size_t chunk_size = (num_queries + num_threads - 1) / num_threads;

    auto process_chunk = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            const std::string& query = queries[i];
            std::unordered_map<int, int> reference_scores; // Reference index to count of matching k-mers

            // Extract k-mers from query
            std::vector<std::string> query_kmers = extract_kmers(query, KMER_LENGTH);
            for (const auto& kmer : query_kmers) {
                auto it = kmer_index.find(kmer);
                if (it != kmer_index.end()) {
                    for (const auto& ref_idx : it->second) {
                        reference_scores[ref_idx]++;
                    }
                }
            }

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
            for (const auto& [ref_idx, count] : reference_scores) {
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
            for (const auto& [ref_idx, count] : reference_scores) {
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

            // Perform Smith-Waterman alignment with each candidate and store the alignments
            for (const auto& [ref_idx, count] : sorted_refs) {
                const auto& reference = references[ref_idx];
                AlignmentResult alignment = smith_waterman(query, reference.second, reference.first);

                // Only consider alignments with percent identity >= 80%
                if (alignment.percent_identity >= 80.0) {
                    qr.top_alignments.push_back(alignment);
                }
            }

            // Store the result
            results[i] = qr;
        }
    };

    // Initialize results vector with default QueryResult
    results.resize(num_queries);

    // Launch threads
    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = std::min(start + chunk_size, num_queries);
        if (start >= end) break; // No more queries to process
        threads.emplace_back(process_chunk, start, end);
    }

    // Join threads
    for (auto& thread : threads) {
        thread.join();
    }
}

int main() {
    try {
        // File paths (adjust these paths as necessary)
        std::string references_file = "/content/800_proj3db.txt";
        std::string queries_file = "/content/800_proj3query_latest.txt";

        // Load references and queries
        std::vector<std::pair<std::string, std::string>> references = load_references(references_file);
        std::vector<std::string> queries = load_queries(queries_file); // Load all queries

        std::cout << "Number of queries loaded: " << queries.size() << "\n";

        if (references.empty()) {
            std::cerr << "Error: No references loaded. Please check your references file.\n";
            return 1;
        }

        if (queries.empty()) {
            std::cerr << "Error: No queries loaded. Please check your queries file.\n";
            return 1;
        }

        // Build k-mer index
        std::unordered_map<std::string, std::vector<int>> kmer_index;
        build_kmer_index(references, KMER_LENGTH, kmer_index);

        // Prepare a vector to hold results
        std::vector<QueryResult> results;

        // Start timing (only for processing)
        auto start_time = std::chrono::high_resolution_clock::now();

        // Process queries in parallel
        process_queries_in_parallel(queries, kmer_index, references, results);

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
                    //std::cout << "Percent identity = " << std::fixed << std::setprecision(2) << alignment.percent_identity << "%\n";
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

        // Print average candidate references
        std::cout << "Average number of candidate references across all queries: "
                  << std::fixed << std::setprecision(2) << average_candidate_references << "\n\n";

        // Print performance metrics
        std::cout << "Total Execution Time (Processing Only): " << std::fixed << std::setprecision(3) << duration.count() << " seconds\n";
        std::cout << "Throughput: " << std::fixed << std::setprecision(3) << throughput << " queries/second\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << "\n";
        return 1;
    }
}
