#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <set>
#include <map>
#include <cassert>

using namespace std;

struct FPNode {
    int item;
    int count;
    FPNode* parent;
    unordered_map<int, FPNode*> children;
    FPNode* next; // 用于头表的链表

    FPNode(int item, FPNode* parent) : item(item), count(0), parent(parent), next(nullptr) {}
};

struct HeaderEntry {
    int count;
    FPNode* head;
    HeaderEntry() : count(0), head(nullptr) {}
};

class FPTree {
public:
    FPNode* root;
    unordered_map<int, HeaderEntry> header_table;

    FPTree() : root(new FPNode(-1, nullptr)) {}

    ~FPTree() {
        deleteTree(root);
    }

    void deleteTree(FPNode* node) {
        if (!node) return;
        for (auto& pair : node->children) {
            deleteTree(pair.second);
        }
        delete node;
    }

    void insert(const vector<int>& items, int count = 1) {
        FPNode* current = root;
        for (int item : items) {
            if (current->children.find(item) == current->children.end()) {
                current->children[item] = new FPNode(item, current);
                // 更新头表
                if (header_table.find(item) == header_table.end()) {
                    header_table[item] = HeaderEntry();
                }
                FPNode* newNode = current->children[item];
                newNode->next = header_table[item].head;
                header_table[item].head = newNode;
                header_table[item].count += count;
            } else {
                header_table[item].count += count;
            }
            current = current->children[item];
            current->count += count;
        }
    }

    bool empty() const {
        return root->children.empty();
    }

    vector<pair<int, int>> get_header_table_ordered_asc() const {
        vector<pair<int, int>> items;
        for (const auto& entry : header_table) {
            items.emplace_back(entry.first, entry.second.count);
        }
        sort(items.begin(), items.end(),
            [](const pair<int, int>& a, const pair<int, int>& b) {
                if (a.second != b.second) return a.second < b.second;
                else return a.first > b.first;
            });
        return items;
    }

    FPNode* get_header_node(int item) const {
        auto it = header_table.find(item);
        if (it != header_table.end()) {
            return it->second.head;
        }
        return nullptr;
    }
};

void minePatterns(const FPTree& tree, const vector<int>& suffix, int suffix_support, int min_support_count,
                  vector<pair<vector<int>, int>>& frequent_patterns) {

    if (!suffix.empty()) {
        frequent_patterns.emplace_back(suffix, suffix_support);
    }

    auto header_items = tree.get_header_table_ordered_asc();
    for (const auto& item_entry : header_items) {
        int item = item_entry.first;
        int support = item_entry.second;

        if (support < min_support_count) {
            continue;
        }

        vector<int> new_suffix;
        new_suffix.push_back(item);
        new_suffix.insert(new_suffix.end(), suffix.begin(), suffix.end());

        frequent_patterns.emplace_back(new_suffix, support);

        vector<pair<vector<int>, int>> conditional_patterns;
        FPNode* node = tree.get_header_node(item);
        while (node != nullptr) {
            vector<int> path;
            FPNode* parent = node->parent;
            while (parent && parent->item != -1) {
                path.push_back(parent->item);
                parent = parent->parent;
            }
            reverse(path.begin(), path.end());
            if (!path.empty()) {
                conditional_patterns.emplace_back(path, node->count);
            }
            node = node->next;
        }

        unordered_map<int, int> item_counts;
        for (const auto& [path, cnt] : conditional_patterns) {
            for (int path_item : path) {
                item_counts[path_item] += cnt;
            }
        }

        vector<int> cond_frequent_items;
        for (const auto& [path_item, total] : item_counts) {
            if (total >= min_support_count) {
                cond_frequent_items.push_back(path_item);
            }
        }

        sort(cond_frequent_items.begin(), cond_frequent_items.end(),
            [&item_counts](int a, int b) {
                if (item_counts[a] != item_counts[b]) return item_counts[a] > item_counts[b];
                else return a < b;
            });

        FPTree cond_tree;
        for (const auto& [path, cnt] : conditional_patterns) {
            vector<int> filtered_path;
            for (int path_item : path) {
                if (find(cond_frequent_items.begin(), cond_frequent_items.end(), path_item) != cond_frequent_items.end()) {
                    filtered_path.push_back(path_item);
                }
            }
            vector<int> sorted_path;
            for (int item : cond_frequent_items) {
                if (find(filtered_path.begin(), filtered_path.end(), item) != filtered_path.end()) {
                    sorted_path.push_back(item);
                }
            }
            if (!sorted_path.empty()) {
                cond_tree.insert(sorted_path, cnt);
            }
        }

        if (!cond_tree.empty()) {
            minePatterns(cond_tree, new_suffix, support, min_support_count, frequent_patterns);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " [min_support] [input_file] [output_file]" << endl;
        return 1;
    }

    double min_support = stod(argv[1]);
    string input_filename = argv[2];
    string output_filename = argv[3];

    vector<vector<int>> transactions;
    unordered_map<int, int> global_item_counts;
    int N = 0;

    ifstream infile(input_filename);
    string line;
    while (getline(infile, line)) {
        vector<int> trans;
        stringstream ss(line);
        string item_str;
        while (getline(ss, item_str, ',')) {
            int item = stoi(item_str);
            trans.push_back(item);
            global_item_counts[item]++;
        }
        if (!trans.empty()) {
            transactions.push_back(trans);
            N++;
        }
    }
    infile.close();

    if (N == 0) {
        ofstream outfile(output_filename);
        outfile.close();
        return 0;
    }

    int min_support_count = ceil(min_support * N - 1e-9);

    vector<int> frequent_items;
    for (const auto& [item, cnt] : global_item_counts) {
        if (cnt >= min_support_count) {
            frequent_items.push_back(item);
        }
    }

    sort(frequent_items.begin(), frequent_items.end(),
        [&global_item_counts](int a, int b) {
            if (global_item_counts[a] != global_item_counts[b]) return global_item_counts[a] > global_item_counts[b];
            else return a < b;
        });

    vector<vector<int>> filtered_transactions;
    for (const auto& trans : transactions) {
        vector<int> filtered;
        for (int item : trans) {
            if (find(frequent_items.begin(), frequent_items.end(), item) != frequent_items.end()) {
                filtered.push_back(item);
            }
        }
        sort(filtered.begin(), filtered.end(),
            [&frequent_items](int a, int b) {
                auto it_a = find(frequent_items.begin(), frequent_items.end(), a);
                auto it_b = find(frequent_items.begin(), frequent_items.end(), b);
                return distance(frequent_items.begin(), it_a) < distance(frequent_items.begin(), it_b);
            });
        if (!filtered.empty()) {
            filtered_transactions.push_back(filtered);
        }
    }

    FPTree fp_tree;
    for (const auto& trans : filtered_transactions) {
        fp_tree.insert(trans, 1);
    }

    vector<pair<vector<int>, int>> frequent_patterns;
    minePatterns(fp_tree, {}, 0, min_support_count, frequent_patterns);

    map<vector<int>, int> unique_patterns;
    for (const auto& [pattern, cnt] : frequent_patterns) {
        vector<int> sorted_pattern = pattern;
        sort(sorted_pattern.begin(), sorted_pattern.end());
        if (unique_patterns.find(sorted_pattern) == unique_patterns.end()) {
            unique_patterns[sorted_pattern] = cnt;
        }
    }

    ofstream outfile(output_filename);
    for (const auto& [pattern, cnt] : unique_patterns) {
        string pattern_str;
        for (size_t i = 0; i < pattern.size(); ++i) {
            if (i > 0) pattern_str += ",";
            pattern_str += to_string(pattern[i]);
        }
        // round to 4 decimal places
        double support = static_cast<double>(cnt) / N;
        support = round(support * 10000.0) / 10000.0;
        outfile << pattern_str << ":" << fixed << setprecision(4) << support << "\n";
    }
    outfile.close();

    return 0;
}