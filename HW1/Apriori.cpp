#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <sstream>
#include <iomanip>
#include <algorithm>

using namespace std;

// 讀取交易紀錄
vector<set<int>> load_transactions(const string &filename) {
    vector<set<int>> transactions;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        set<int> transaction;
        stringstream ss(line);
        string item;
        while (getline(ss, item, ',')) {
            transaction.insert(stoi(item));
        }
        transactions.push_back(transaction);
    }

    file.close();
    return transactions;
}

// 計算頻繁的 1-itemset
map<int, double> get_frequent_1_itemsets(const vector<set<int>> &transactions, double min_support) {
    map<int, int> item_counts;
    int total_transactions = transactions.size();

    for (const auto &transaction : transactions) {
        for (int item : transaction) {
            item_counts[item]++;
        }
    }

    map<int, double> frequent_items;
    for (const auto &[item, count] : item_counts) {
        double support = (double)count / total_transactions;
        if (support >= min_support) {
            frequent_items[item] = support;
        }
    }

    return frequent_items;
}

// 產生候選 k-itemsets
set<set<int>> generate_candidates(const set<set<int>> &previous_frequent_itemsets, int k) {
    set<set<int>> candidates;
    vector<set<int>> prev_items(previous_frequent_itemsets.begin(), previous_frequent_itemsets.end());

    for (size_t i = 0; i < prev_items.size(); i++) {
        for (size_t j = i + 1; j < prev_items.size(); j++) {
            set<int> candidate = prev_items[i];
            candidate.insert(prev_items[j].begin(), prev_items[j].end());

            // **確保新產生的候選集大小為 k**
            if (candidate.size() == k) {
                // **檢查所有 (k-1) 子集是否都在 Lk-1**
                bool all_subsets_frequent = true;
                for (int item : candidate) {
                    set<int> subset = candidate;
                    subset.erase(item);
                    if (previous_frequent_itemsets.find(subset) == previous_frequent_itemsets.end()) {
                        all_subsets_frequent = false;
                        break;
                    }
                }
                if (all_subsets_frequent) {
                    candidates.insert(candidate);
                }
            }
        }
    }
    return candidates;
}

// 過濾頻繁項目集
map<set<int>, double> filter_frequent_itemsets(
    const set<set<int>> &candidates,
    const vector<set<int>> &transactions,
    double min_support) {

    map<set<int>, int> itemset_counts;
    int total_transactions = transactions.size();

    for (const auto &transaction : transactions) {
        for (const auto &candidate : candidates) {
            // **檢查候選項目是否在交易中**
            if (includes(transaction.begin(), transaction.end(), candidate.begin(), candidate.end())) {
                itemset_counts[candidate]++;
            }
        }
    }

    // **計算支持度並過濾**
    map<set<int>, double> frequent_itemsets;
    for (const auto &[itemset, count] : itemset_counts) {
        double support = (double)count / total_transactions;
        if (support >= min_support) {
            frequent_itemsets[itemset] = support;
        }
    }

    return frequent_itemsets;
}

// Apriori 演算法
    map<set<int>, double> apriori(vector<set<int>> &transactions, double min_support) {
        map<set<int>, double> all_frequent_itemsets;
    
        // **計算 L1**
        map<int, double> L1 = get_frequent_1_itemsets(transactions, min_support);
        set<set<int>> current_L;
        for (const auto &[item, support] : L1) {
            current_L.insert({item});
            all_frequent_itemsets[{{item}}] = support;
        }
    
        int k = 2;
        while (!current_L.empty()) {
            // **產生候選項目集 Ck**
            set<set<int>> candidates = generate_candidates(current_L, k);
    
            // **計算支持度，過濾出 Lk**
            map<set<int>, double> frequent_itemsets = filter_frequent_itemsets(candidates, transactions, min_support);
    
            current_L.clear();
            for (const auto &[itemset, support] : frequent_itemsets) {
                all_frequent_itemsets[itemset] = support;
                current_L.insert(itemset);
            }
    
            k++;
        }
    
        return all_frequent_itemsets;
    }
    

// **排序後輸出結果**
void save_results(const map<set<int>, double> &frequent_patterns, const string &output_filename) {
    // **轉換 map 到 vector，方便排序**
    vector<pair<set<int>, double>> sorted_patterns(frequent_patterns.begin(), frequent_patterns.end());

    // **排序 (按照 support 值由大到小排列)**
    sort(sorted_patterns.begin(), sorted_patterns.end(), [](const auto &a, const auto &b) {
        return a.second > b.second;  // 降序排列 (greater)
    });

    // **輸出結果**
    ofstream file(output_filename);
    
    for (const auto &[itemset, support] : sorted_patterns) {
        vector<int> items(itemset.begin(), itemset.end());
        sort(items.begin(), items.end());  // 確保 itemset 內的項目有序
        
        for (size_t i = 0; i < items.size(); i++) {
            if (i > 0) file << ",";
            file << items[i];
        }
        file << ":" << fixed << setprecision(4) << support << endl;
    }

    file.close();
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        cerr << "使用方式: ./你的學號_hw1 [min_support] [輸入檔案] [輸出檔案]" << endl;
        return 1;
    }

    double min_support = stod(argv[1]);
    string input_file = argv[2];
    string output_file = argv[3];

    vector<set<int>> transactions = load_transactions(input_file);
    map<set<int>, double> frequent_patterns = apriori(transactions, min_support);
    save_results(frequent_patterns, output_file);

    return 0;
}
