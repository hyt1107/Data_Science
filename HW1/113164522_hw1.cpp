#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <map>
#include <cassert>
#include <set>
#include <functional>


using namespace std;


struct FPNode {
    int item;
    int count;
    FPNode* parent;
    // 用 item -> 子節點的映射
    unordered_map<int, FPNode*> children;
    // 用於頭表的鏈表指標
    FPNode* next;

    FPNode(int item, FPNode* parent) : item(item), count(0), parent(parent), next(nullptr) {}
};

struct HeaderEntry {
    int total_count;
    FPNode* head;
    HeaderEntry() : total_count(0), head(nullptr) {}
};

// --------------- FP-tree 結構 ---------------
class FPTree {
public:
    FPNode* root;
    // 對每個 item, 紀錄它的 total_count 以及在樹中串聯的起始節點
    unordered_map<int, HeaderEntry> header_table;

    FPTree() {
        root = new FPNode(-1, nullptr);
    }

    ~FPTree() {
        deleteTree(root);
    }

    // 插入一筆已經根據全局索引排序好的交易，計數 = count
    void insert(const vector<int>& items, int count=1) {
        FPNode* current = root;
        for (int item : items) {
            // 若不存在子節點就新建
            if (current->children.find(item) == current->children.end()) {
                FPNode* newNode = new FPNode(item, current);
                current->children[item] = newNode;
                // 更新 header_table
                if (header_table.find(item) == header_table.end()) {
                    header_table[item] = HeaderEntry();
                }
                newNode->next = header_table[item].head;
                header_table[item].head = newNode;
            }
            // 無論新建與否, 更新當前 item 在 header 裏的 total_count
            header_table[item].total_count += count;
            // 移動到下一層
            current = current->children[item];
            current->count += count;
        }
    }

    bool empty() const {
        return (root->children.empty());
    }

private:
    void deleteTree(FPNode* node) {
        if (!node) return;
        for (auto &ch : node->children) {
            deleteTree(ch.second);
        }
        delete node;
    }
};

// ------------------ 全域方法 -------------------

// 檢查某棵 FP-tree 是否僅剩一條路徑
bool isSinglePath(FPNode* node) {
    // 空或葉節點 視為單一路徑
    if (!node) return true;
    // 若某個節點的子節點 > 1, 就不是單一路徑
    if (node->children.size() > 1) return false;
    // 對唯一子節點做遞迴檢查
    if (node->children.size() == 1) {
        // 取出子節點 ( map.begin()->second )
        return isSinglePath(node->children.begin()->second);
    }
    // 沒有子節點 -> 到葉子
    return true;
}

// 將單一路徑上所有項目收集下來 (從 root 底下開始), 回傳 (item, count) 的序列
// 注意：每個節點都有自己的 count，代表"從 root 到該節點" 這個前綴的次數
//       單一路徑時，往下的節點 count 不一定都是一樣，但最小值就是最末端葉子的 count。
void collectSinglePathItems(FPNode* node, vector<pair<int,int>>& path) {
    // 走到底就結束
    if (!node) return;
    // root->item == -1 表示虛擬節點
    // 只要不是 root，就加入
    if (node->item != -1) {
        // node->count 表示 (root ~ node) 這條前綴的總出現次數
        path.emplace_back(node->item, node->count);
    }
    // 繼續往下 (可能沒有子節點 or 只有1個)
    if (node->children.size() == 1) {
        collectSinglePathItems(node->children.begin()->second, path);
    } 
    // 若是 0 個子節點，就到了葉子；>1 個子節點就不會被呼叫到這邊(事先檢查 isSinglePath)
}

// 針對一條單一路徑, 產生所有子集合 (除了空集合), 並把支援度記錄到 frequent_patterns
// suffix 是已經挖掘到的後綴模式, suffix_support 是該後綴的支援度
// path: [(item1, cnt1), (item2, cnt2), ...]，單一路徑從上到下
// 整個子集合對應的support = min{cnt_i} (對於該子集合包含的最底層 item)
void generateSubsetsFromSinglePath(const vector<pair<int,int>>& path,
                                   const vector<int>& suffix,
                                   int suffix_support,
                                   int min_support_count,
                                   vector<pair<vector<int>, int>>& frequent_patterns) 
{
    // path.size() = k, 共有 2^k - 1 種非空子集合
    // 但通常會用遞迴或位元枚舉法來產生
    // 這裡示範簡單的遞迴子集合生成

    // 先擷取 path 中的 items
    vector<int> items;
    vector<int> counts;
    items.reserve(path.size());
    counts.reserve(path.size());
    for (auto &p : path) {
        items.push_back(p.first);
        counts.push_back(p.second);
    }

    // 遞迴: 將子集合 push_back 到 frequent_patterns
    // 為了找該子集合的支援度，我們要看該子集合裡最後(最深)那個 item 的 count 的最小值
    
    // 這裡用 backtracking, 每次考慮是否加入當前 item
    vector<int> current;
    function<void(int, int)> dfs = [&](int idx, int current_min) {
        if (idx == (int)items.size()) {
            // 走到盡頭, current 若非空 -> 加入
            if (!current.empty()) {
                // suffix + current 就是一個完整模式
                vector<int> pattern = current;
                // 先把 suffix 放最右邊 (題主程式是後綴在右？這裡隨意，也可 concat)
                pattern.insert(pattern.end(), suffix.begin(), suffix.end());
                // 加入結果
                frequent_patterns.push_back({pattern, current_min});
            }
            return;
        }

        // 選擇「不加入」 idx
        dfs(idx+1, current_min);

        // 選擇「加入」 idx
        current.push_back(items[idx]);
        int new_min = current_min;
        // 與 idx 位置的 count 取 min
        if (new_min == 0) new_min = counts[idx];
        else new_min = min(new_min, counts[idx]);
        dfs(idx+1, new_min);
        current.pop_back();
    };

    // suffix_support 可能是0 (表示還沒定義), 之後會在遞迴中計算
    dfs(0, suffix_support);
}

// 遞迴挖掘 FP-tree 的主要函式
// suffix, suffix_support 用來記錄「到目前為止」已確定的模式與其支持度
// frequent_patterns 負責收集最終結果
void minePatterns(FPTree& tree,
                  const vector<int>& suffix,
                  int suffix_support,
                  int min_support_count,
                  vector<pair<vector<int>, int>>& frequent_patterns,
                  const vector<int>& global_order) 
{
    // 若 suffix 非空, 可直接將 (suffix, suffix_support) 放入結果
    // 不過部分實作會把加入結果的時機放在 loop 時
    if (!suffix.empty() && suffix_support > 0) {
        frequent_patterns.push_back({suffix, suffix_support});
    }

    // 先檢查頭表中的項目 (這裡用 global_order 來確定遍歷順序)
    // 也可以只遍歷出現在 header_table 的 item, 排序依舊按照 global_order
    vector<int> candidate_items;
    candidate_items.reserve(global_order.size());
    for (int it : global_order) {
        auto itH = tree.header_table.find(it);
        if (itH != tree.header_table.end()) {
            if (itH->second.total_count >= min_support_count) {
                candidate_items.push_back(it);
            }
        }
    }
    if (candidate_items.empty()) return;

    // 對每個候選 item 建構條件 FP-tree
    for (int item : candidate_items) {
        int item_support = tree.header_table[item].total_count;
        if (item_support < min_support_count) {
            continue;
        }

        // 形成新的後綴 new_suffix
        // FP-Growth 通常會把 item 放在「suffix 的左邊或右邊」，視實作而定
        vector<int> new_suffix = suffix;
        new_suffix.push_back(item);

        // 先把 new_suffix (對應 support) 加入結果
        // 這裡的支持度是 item 在當前樹中的 total_count
        frequent_patterns.push_back({new_suffix, item_support});

        // 收集條件模式基底
        vector<pair<vector<int>, int>> conditional_patterns;
        FPNode* node = tree.header_table[item].head;
        while (node != nullptr) {
            int path_count = node->count;  // 這條路徑出現多少次
            vector<int> path;
            FPNode* parent = node->parent;
            // 回溯到 root(-1) 為止
            while (parent && parent->item != -1) {
                path.push_back(parent->item);
                parent = parent->parent;
            }
            // 逆序 -> 轉為根到葉方向
            reverse(path.begin(), path.end());
            if (!path.empty()) {
                conditional_patterns.push_back({path, path_count});
            }
            node = node->next;
        }

        // 統計在 conditional_patterns 中各項目的次數
        unordered_map<int,int> cond_count;
        for (auto &cp : conditional_patterns) {
            auto &p = cp.first;
            int c = cp.second;
            for (int x : p) {
                cond_count[x] += c;
            }
        }

        // 建構條件 FP-tree
        FPTree cond_tree;
        // 對於每條路徑, 過濾掉 cond_count < min_support_count 的項目
        // 並依 global_order 順序插入
        for (auto &cp : conditional_patterns) {
            auto &p = cp.first;
            int c = cp.second;

            // 篩選 + 依 global_order 的次序重排
            vector<int> filtered;
            filtered.reserve(p.size());
            for (int x : p) {
                if (cond_count[x] >= min_support_count) {
                    filtered.push_back(x);
                }
            }
            if (filtered.empty()) continue;

            // 按 global_order 排序 (用 item->index 之類的方式)
            // 這裡假設 global_order.size() 不會太大; 也可用兩指標法
            sort(filtered.begin(), filtered.end(), [&](int a, int b){
                // 用在 global_order 的位置
                // 由於 global_order 是 item(0)~item(n-1), 直接做查找也可用雜湊
                // 簡化示範：直接比較 a,b 在 global_order 的索引
                auto ia = find(global_order.begin(), global_order.end(), a);
                auto ib = find(global_order.begin(), global_order.end(), b);
                return (ia < ib);
            });

            cond_tree.insert(filtered, c);
        }

        // 若條件 FP-tree 不為空，就遞迴挖掘
        if (!cond_tree.empty()) {
            // 單一路徑偵測
            if (isSinglePath(cond_tree.root)) {
                // 收集整條路徑
                vector<pair<int,int>> path_items;
                collectSinglePathItems(cond_tree.root, path_items);
                // 直接一次性枚舉所有子集合
                generateSubsetsFromSinglePath(path_items, new_suffix, 0, 
                                              min_support_count, frequent_patterns);
            } else {
                // 正常遞迴
                minePatterns(cond_tree, new_suffix, item_support,
                             min_support_count, frequent_patterns, global_order);
            }
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

    // 讀取交易
    ifstream infile(input_filename);
    if (!infile.is_open()) {
        cerr << "Cannot open input file.\n";
        return 1;
    }

    vector<vector<int>> transactions;
    unordered_map<int, int> global_item_counts;
    int N = 0;

    string line;
    while (getline(infile, line)) {
        if (line.empty()) continue;
        vector<int> trans;
        stringstream ss(line);
        string token;
        while (getline(ss, token, ',')) {
            int item = stoi(token);
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
        // 沒交易，直接輸出空檔
        ofstream outfile(output_filename);
        outfile.close();
        return 0;
    }

    // 計算 min_support_count
    int min_support_count = (int)ceil(min_support * N - 1e-9);

    // 篩選全局頻繁項目
    vector<int> frequent_items;
    frequent_items.reserve(global_item_counts.size());
    for (auto &kv : global_item_counts) {
        if (kv.second >= min_support_count) {
            frequent_items.push_back(kv.first);
        }
    }
    // 按「支援度由大到小；若相同則 item 由小到大」排序
    sort(frequent_items.begin(), frequent_items.end(), [&](int a, int b){
        if (global_item_counts[a] != global_item_counts[b]) {
            return global_item_counts[a] > global_item_counts[b];
        }
        return a < b;
    });

    // 建立 item -> index 的映射 (global_order 就是 frequent_items，但我們這裡用"index"找更快)
    unordered_map<int,int> item_to_idx;
    item_to_idx.reserve(frequent_items.size());
    for (int i=0; i<(int)frequent_items.size(); i++) {
        item_to_idx[frequent_items[i]] = i; 
    }

    // 依照「全局索引小->大」的順序，用一個 global_order 做挖掘時的遍歷基準
    // 如果要保留「支援度大->小」，可以在遞迴時反向讀這個 global_order
    // 這裡為示範, 直接用 "frequent_items" 當作 global_order (內部是大->小)
    // 也可做一次 reverse 或自行調整
    vector<int> global_order = frequent_items;

    // 過濾交易 + 依全局索引排序(小->大) [或保留相同順序, 依實際需求]
    vector<vector<int>> filtered_transactions;
    filtered_transactions.reserve(transactions.size());
    for (auto &trans : transactions) {
        // 先過濾
        vector<int> temp;
        temp.reserve(trans.size());
        for (int it : trans) {
            auto itFind = item_to_idx.find(it);
            if (itFind != item_to_idx.end()) {
                temp.push_back(it);
            }
        }
        if (temp.empty()) continue;
        // 接著按照 item_to_idx[it] 由小到大排序
        sort(temp.begin(), temp.end(), [&](int a, int b){
            return item_to_idx[a] < item_to_idx[b];
        });
        filtered_transactions.push_back(temp);
    }

    // 建構 FP-tree
    FPTree fp_tree;
    for (auto &trans : filtered_transactions) {
        fp_tree.insert(trans, 1);
    }

    // 開始挖掘
    vector<pair<vector<int>, int>> frequent_patterns;
    minePatterns(fp_tree, {}, 0, min_support_count, frequent_patterns, global_order);

    // 去重 (有些模式可能重覆)
    // 用 map 以「排序後的 pattern」為 key
    map<vector<int>, int> unique_patterns;
    for (auto &p : frequent_patterns) {
        vector<int> sorted_pattern = p.first;
        sort(sorted_pattern.begin(), sorted_pattern.end());
        // 如果之前沒紀錄，或這次支援度更大，就更新
        // (通常 FP-growth 產生同個模式時支援度應該一致，若不一致可視情況取最大或最小)
        if (unique_patterns.find(sorted_pattern) == unique_patterns.end()) {
            unique_patterns[sorted_pattern] = p.second;
        } else {
            // 也可比較取較大支援度
            if (p.second > unique_patterns[sorted_pattern]) {
                unique_patterns[sorted_pattern] = p.second;
            }
        }
    }

    // 輸出到檔案
    ofstream outfile(output_filename);
    if (!outfile.is_open()) {
        cerr << "Cannot open output file.\n";
        return 1;
    }
    for (auto &kv : unique_patterns) {
        auto &pattern = kv.first;
        int cnt = kv.second;
        // pattern 以逗號連接
        string pattern_str;
        for (size_t i = 0; i < pattern.size(); ++i) {
            if (i > 0) pattern_str += ",";
            pattern_str += to_string(pattern[i]);
        }
        double support = (double)cnt / (double)N;
        support = floor(support * 10000 + 0.5) / 10000.0; // 四捨五入至4位小數
        outfile << pattern_str << ":" << fixed << setprecision(4) << support << "\n";
    }
    outfile.close();

    return 0;
}
