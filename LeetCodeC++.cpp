#include <iostream>
#include <string>
#include <string.h>
#include <cstdio>
#include <math.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <tchar.h>
#include <stack>
#include <numeric>
using namespace std;

// 1. 两数之和
class Solution1 {
public:
	/*vector<int> twoSum(vector<int>& nums, int target) {
		vector<int> res;
		int len = nums.size();
		for (int i = 0; i < len; i++) {
			for (int j = i + 1; j < len; j++) {
				if (nums[i] + nums[j] == target) {
					res.push_back(i);
					res.push_back(j);
					return res;
				}
			}
		}
		return res;
	}*/
	vector<int> twoSum(vector<int>& nums, int target) {
		vector<int> res;
		map<int, int> m;
		int len = nums.size();
		for (int i = 0; i < len; i++) {
			int need_value = target - nums[i];
			if (m.find(need_value) != m.end()) {
				res.push_back(m.find(need_value)->second);
				res.push_back(i);
				break;
			}
			m.insert(make_pair(nums[i], i));
		}
		return res;
	}
};

// 2. 两数相加
class Solution2 {
	struct ListNode {
		int val;
		ListNode *next;
		ListNode(int x) : val(x), next(NULL) {}
	};
public:
	ListNode * addTwoNumbers(ListNode* l1, ListNode* l2) {
		ListNode *l3 = new ListNode(0);
		ListNode *p = l1, *q = l2, *curr = l3;
		int carry = 0;
		while (p != NULL || q != NULL) {
			int x = (p != NULL ? p->val : 0);
			int y = (q != NULL ? q->val : 0);
			int sum = carry + x + y;
			carry = sum / 10;
			curr->next = new ListNode(sum % 10);
			if (p != NULL) p = p->next;
			if (q != NULL) q = q->next;
			curr = curr->next;
		}
		if (carry != 0) {
			curr->next = new ListNode(1);
		}
		return l3->next;
	}
};

// 3. 无重复字符的最长子串
class Solution3 {
public:
	/*bool hasDuplicates(string s) {
		int len = s.length();
		if (len == 1) return false;
		for (int i = 0; i < len - 1; i++) {
			if (s[i] == s[len - 1]) {
				return true;
			}
		}
		return false;
	}
	int lengthOfLongestSubstring(string s) {
		string curr = "";
		int max_len = 0;
		int s_len = s.length();
		for (int i = 0; i < s_len; i++) {
			curr = curr + s[i];
			while (hasDuplicates(curr)) {
				curr.erase(curr.begin());
			}
			max_len = max_len > curr.length() ? max_len : curr.length();
		}
		return max_len;
	}*/
	int lengthOfLongestSubstring(string s) {
		set<char> ans_set;
		int i = 0, j = 0, max_len = 0;
		int s_len = s.length();
		while (i < s_len && j < s_len) {
			if (ans_set.find(s[j]) != ans_set.end()) {
				ans_set.erase(s[i++]);
			}
			else {
				ans_set.insert(s[j++]);
				max_len = max_len > (j - i) ? max_len : (j - i);
			}
		}
		return max_len;
	}
};

// 4. 两个排序数组的中位数
class Solution4 {
public:
	/*double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
		int length1 = nums1.size();
		int length2 = nums2.size();
		int length_all = length1 + length2;
		int *all = new int[length_all];
		int i = 0, j = 0, k = 0;
		while (i < length1 && j < length2) {
			if (nums1[i] < nums2[j]) all[k] = nums1[i++];
			else all[k] = nums2[j++];
			k++;
		}
		while (i < length1) all[k++] = nums1[i++];
		while (j < length2) all[k++] = nums2[j++];

		double result;
		if (length_all % 2 == 0) result = (all[length_all / 2 - 1] + all[length_all / 2])*1.0 / 2;
		else result = all[length_all / 2];
		return result;
	}*/
	int getKthMin(vector<int>& a, int a_start, vector<int>& b, int b_start, int k) {
		int a_len = a.size(), b_len = b.size();
		if (a_start >= a_len) return b[b_start + k - 1];
		if (b_start >= b_len) return a[a_start + k - 1];
		if (k == 1) return a[a_start] < b[b_start] ? a[a_start] : b[b_start];
		int a_min = INT_MAX, b_min = INT_MAX;
		if (a_start + k / 2 - 1 < a_len) a_min = a[a_start + k / 2 - 1];
		if (b_start + k / 2 - 1 < b_len) b_min = b[b_start + k / 2 - 1];
		return a_min < b_min ? getKthMin(a, a_start + k / 2, b, b_start, k - k / 2) : getKthMin(a, a_start, b, b_start + k / 2, k - k / 2);
	}
	double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
		int length1 = nums1.size();
		int length2 = nums2.size();
		int length_all = length1 + length2;
		int left = (length_all + 1) / 2;
		int right = (length_all + 2) / 2;
		return (getKthMin(nums1, 0, nums2, 0, left) + getKthMin(nums1, 0, nums2, 0, right)) * 1.0 / 2;
	}
};

// 5. 最长回文子串
class Solution5 {
public:
	int expandAroundCenter(string s, int left, int right) {
		int l = left, r = right, len = s.length();
		while (l >= 0 && r < len && s[l] == s[r]) {
			l--;
			r++;
		}
		return r - l - 1;
	}
	string longestPalindrome(string s) {
		int start = 0, end = 0;
		int len = s.length();
		for (int i = 0; i < len; i++) {
			int len1 = expandAroundCenter(s, i, i);
			int len2 = expandAroundCenter(s, i, i + 1);
			int len_max = len1 > len2 ? len1 : len2;
			if (len_max > end - start) {
				start = i - (len_max - 1) / 2;
				end = i + len_max / 2;
			}
		}
		return s.substr(start, end - start + 1);
	}
};

// 6. Z字形变换
class Solution6 {
public:
	string convert(string s, int numRows) {
		if (numRows == 1) return s;
		int min_row = numRows < int(s.size()) ? numRows : int(s.size());
		vector<string> rows(min_row);
		int curr_row = 0;
		bool goingDown = false;
		for (char c : s) {
			rows[curr_row] += c;
			if (curr_row == 0 || curr_row == numRows - 1) {
				goingDown = !goingDown;
			}
			curr_row += (goingDown ? 1 : -1);
		}
		string res;
		for (string row : rows) {
			res += row;
		}
		return res;
	}
};

// 7. 反转整数
class Solution7 {
public:
	/*int reverse(int x) {
		long long result = 0;
		while (x != 0) {
			int temp = x % 10;
			result = result * 10 + temp;
			x /= 10;
		}
		if (result > INT_MAX || result < INT_MIN) {
			result = 0;
		}
		return result;
	}*/
	int reverse(int x) {
		int result = 0;
		while (x != 0) {
			int pop = x % 10;
			x /= 10;
			if (result > INT_MAX / 10 || (result == INT_MAX / 10 && pop > 7)) return 0;
			if (result < INT_MIN / 10 || (result == INT_MIN / 10 && pop < -8)) return 0;
			result = 10 * result + pop;
		}
		return result;
	}
};

// 8. 字符串转整数 (atoi)
class Solution8 {
public:
	/*int str_to_int(string str) {
		int len = str.length();
		int ret = 0, minus = 0, flag = 0;
		for (int i = 0; i < len; i++) {
			if (str[i] == '-') {
				minus = 1; continue;
			}
			if (ret == INT_MIN / 10 && str[i] >= '8' || ret < INT_MIN / 10 ||
				ret == INT_MAX / 10 && str[i] <= '7' || ret < INT_MAX / 10) {
				ret = ret * 10 + (str[i] - '0');
			}
			else {
				ret = (minus == 1 ? INT_MIN : INT_MAX);
				flag = 1;
				break;
			}
		}
		if (!flag && minus) ret = -ret;
		return ret;
	}

	int myAtoi(string str) {
		string str_num;
		int len = str.length();
		int minus = 0, has_num = 0, valid = 0, has_symbol = 0;
		for (int i = 0; i < len; i++) {
			if (str[i] == ' ') {
				if (valid) break;
				continue;
			}
			if (!valid && !has_symbol && str[i] == '+') {
				valid = 1; has_symbol = 1; continue;
			}
			if (!valid && !has_symbol && !minus && str[i] == '-') {
				valid = 1; has_symbol = 1; minus = 1; continue;
			}
			if (str[i] < '0' || str[i] > '9') break;
			has_num = 1;
			valid = 1;
			str_num += str[i];
		}
		if (minus) str_num = "-" + str_num;
		if (!has_num) str_num = "0";
		int ret = str_to_int(str_num);
		return ret;
	}*/
	int myAtoi(string str) {
		if (str.empty()) return 0;
		int sign = 1, base = 0, i = 0, n = str.size();
		while (i < n && str[i] == ' ') ++i;
		if (str[i] == '+' || str[i] == '-') {
			sign = (str[i++] == '+') ? 1 : -1;
		}
		while (i < n && str[i] >= '0' && str[i] <= '9') {
			if (base > INT_MAX / 10 || (base == INT_MAX / 10 && str[i] - '0' > 7)) {
				return (sign == 1) ? INT_MAX : INT_MIN;
			}
			base = 10 * base + (str[i++] - '0');
		}
		return base * sign;
	}
};

// 9. 回文数
class Solution9 {
public:
	/*bool isPalindrome(int x) {
		if (x < 0) return false;
		char *str = new char[20];
		sprintf(str, "%d", x);
		int len = strlen(str);
		for (int i = 0; i <= len / 2; i++) {
			if (str[i] != str[len - i - 1]) {
				return false;
			}
		}
		return true;
	}*/
	bool isPalindrome(int x) {
		if (x < 0 || (x % 10 == 0 && x != 0)) {
			return false;
		}
		int revertedNumber = 0;
		while (x > revertedNumber) {
			revertedNumber = revertedNumber * 10 + x % 10;
			x /= 10;
		}
		return x == revertedNumber || x == revertedNumber / 10;
	}
};

// 11. 盛最多水的容器
class Solution11 {
public:
	/*int maxArea(vector<int>& height) {
		int len = height.size();
		int max_area = 0;
		for (int i = 0; i < len; i++) {
			for (int j = i + 1; j < len; j++) {
				int area = min(height[j], height[i]) * (j - i);
				max_area = max(max_area, area);
			}
		}
		return max_area;
	}*/
	int maxArea(vector<int>& height) {
		int max_area = 0;
		int left = 0, right = height.size() - 1;
		while (left < right) {
			int area = min(height[left], height[right]) * (right - left);
			max_area = max(max_area, area);
			if (height[left] < height[right]) left++;
			else right--;
		}
		return max_area;
	}
};

// 19. 删除链表的倒数第N个节点
class Solution19 {
	struct ListNode {
		int val;
		ListNode *next;
		ListNode(int x) : val(x), next(NULL) {}
	};
public:
	// 一个指针，遍历两次
	/*ListNode * removeNthFromEnd(ListNode* head, int n) {
		ListNode *first = head;
		int len = 0;
		while (first) {
			len++;
			first = first->next;
		}
		ListNode *new_head = new ListNode(0);
		new_head->next = head;
		first = new_head;
		int after = 1;
		while (after != len - n + 1) {
			first = first->next;
			after++;
		}
		first->next = first->next->next;
		return new_head->next;
	}*/
	// 两个指针，相隔n步，同时出发，遍历一次
	ListNode * removeNthFromEnd(ListNode* head, int n) {
		ListNode *new_head = new ListNode(0);
		new_head->next = head;
		ListNode *first = new_head;
		ListNode *second = new_head;
		for (int i = 0; i < n; i++) {
			second = second->next;
		}
		while (second->next) {
			first = first->next;
			second = second->next;
		}
		first->next = first->next->next;
		return new_head->next;
	}
};

// 21. 合并两个有序链表
class Solution21 {
	struct ListNode {
		int val;
		ListNode *next;
		ListNode(int x) : val(x), next(NULL) {}
	};
public:
	/*ListNode * mergeTwoLists(ListNode* l1, ListNode* l2) {
		ListNode *dummy = new ListNode(-1), *cur = dummy;
		while (l1 && l2) {
			if (l1->val < l2->val) {
				cur->next = l1;
				l1 = l1->next;
			}
			else {
				cur->next = l2;
				l2 = l2->next;
			}
			cur = cur->next;
		}
		cur->next = l1 ? l1 : l2;
		return dummy->next;
	}*/
	// 递归
	ListNode * mergeTwoLists(ListNode* l1, ListNode* l2) {
		if (!l1) return l2;
		if (!l2) return l1;
		if (l1->val < l2->val) {
			l1->next = mergeTwoLists(l1->next, l2);
			return l1;
		}
		else {
			l2->next = mergeTwoLists(l1, l2->next);
			return l2;
		}
	}
};

// 22. 括号生成
class Solution22 {
public:
	// 暴力法
	/*bool valid(string curr) {
		int flag = 0;
		for (int i = 0; i < curr.size(); i++) {
			if (curr[i] == '(') flag++;
			if (curr[i] == ')') flag--;
			if (flag < 0) return false;
		}
		return (flag == 0);
	}
	void generateAll(string curr, int length, vector<string> &res) {
		if (curr.size() == length) {
			if (valid(curr)) {
				res.push_back(curr);
			}
			return;
		}
		generateAll(curr + '(', length, res);
		generateAll(curr + ')', length, res);
	}
	vector<string> generateParenthesis(int n) {
		vector<string> combinations;
		generateAll("", 2 * n, combinations);
		return combinations;
	}*/
	// 回溯法
	void backtrack(string curr, int open, int close, int max, vector<string> &res) {
		if (curr.size() == 2 * max) {
			res.push_back(curr);
			return;
		}
		if (open < max) backtrack(curr + '(', open + 1, close, max, res);
		if (close < open) backtrack(curr + ')', open, close + 1, max, res);
	}
	vector<string> generateParenthesis(int n) {
		vector<string> combinations;
		backtrack("", 0, 0, n, combinations);
		return combinations;
	}
};

// 23. Merge k Sorted Lists
// 题意：合并 k 个有序的链表
class Solution23 {
	struct ListNode {
		int val;
		ListNode *next;
		ListNode(int x) : val(x), next(NULL) {}
	};
	struct cmp {
		bool operator () (ListNode* l1, ListNode* l2) {
			return l1->val > l2->val;
		}
	};
public:
	// 解1：分为 (k + 1) / 2 组，两两合并
	/*ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
		ListNode* head = new ListNode(-1);
		ListNode* cur = head;
		while (l1 && l2) {
			if (l1->val < l2->val) {
				cur->next = l1;
				l1 = l1->next;
			}
			else {
				cur->next = l2;
				l2 = l2->next;
			}
			cur = cur->next;
		}
		if (l1) cur->next = l1;
		if (l2) cur->next = l2;
		return head->next;
	}
	ListNode* mergeKLists(vector<ListNode*>& lists) {
		if (lists.size() == 0) return NULL;
		int n = lists.size();
		while (n > 1) {
			int k = (n + 1) / 2;
			for (int i = 0; i < n / 2; i++) {
				lists[i] = mergeTwoLists(lists[i], lists[i + k]);
			}
			n = k;
		}
		return lists[0];
	}*/

	// 解2：优先队列
	ListNode* mergeKLists(vector<ListNode*>& lists) {
		priority_queue<ListNode*, vector<ListNode*>, cmp> que;
		for (int i = 0; i < lists.size(); i++) {
			if (lists[i]) que.push(lists[i]);
		}
		ListNode *head = NULL, *temp = NULL, *pre = NULL;
		while (!que.empty()) {
			temp = que.top();
			que.pop();
			if (!pre) head = temp;
			else pre->next = temp;
			pre = temp;
			if (temp->next) que.push(temp->next);
		}
		return head;
	}
};

// 26. 删除排序数组中的重复项
class Solution26 {
public:
	/*int removeDuplicates(vector<int>& nums) {
		if (nums.size() == 0) return 0;
		int len = nums.size();
		int new_len = 0;
		for (int i = 0; i < len - 1; ) {
			if (nums[i + 1] == nums[i]) {
				for (int j = i + 1; j < len; j++) {
					nums[j - 1] = nums[j];
				}
				len--;
			}
			else {
				i++; new_len++;
			}
		}
		if (len > 1 && nums[len - 1] != nums[len - 2]) new_len++;
		else if (len == 1) new_len = 1;
		return new_len;
	}*/
	int removeDuplicates(vector<int>& nums) {
		if (nums.size() == 0) return 0;
		int i = 0, len = nums.size();
		for (int j = 1; j < len; j++) {
			if (nums[j] != nums[i]) {
				i++;
				nums[i] = nums[j];
			}
		}
		return i + 1;
	}
};

// 29. 两数相除
class Solution29 {
public:
	// 两数相除，要求不使用乘法、除法和 mod 运算符
	// 使用log，a/b = e^(ln(a))/e^(ln(b)) = e^(ln(a)-ln(b))
	/*int divide(int dividend, int divisor) {
		if (dividend == 0) return 0;
		double t1 = log(fabs(dividend));
		double t2 = log(fabs(divisor));
		long long result = exp(t1 - t2);
		if ((dividend < 0) ^ (divisor < 0)) result = -result;
		if (result > INT_MAX) result = INT_MAX;
		return result;
	}*/
	int divide(int dividend, int divisor) {
		long long result = 0;
		long long m = abs((long long)dividend);
		long long n = abs((long long)divisor);

		// dvd >= 2^k1*dvs + 2^k2*dvs ...
		while (m >= n) {
			long long temp = n, count = 1;
			while (m >= (temp << 1)) {
				temp <<= 1;
				count <<= 1;
			}
			m -= temp;
			result += count;
		}
		if ((dividend > 0) ^ (divisor > 0))  result = -result;
		return result > INT_MAX ? INT_MAX : (int)result;
	}
};

// 31. Next Permutation 下一个排列
class Solution31 {
public:
	/*
	通过观察原数组可以发现，如果从末尾往前看，数字逐渐变大，到了2时才减小的，
	然后我们再从后往前找第一个比2大的数字，是3，那么我们交换2和3，再把此时3后面的所有数字转置一下即可，步骤如下：
	1　　2　　7　　4　　3　　1
	1　　3　　7　　4　　2　　1
	1　　3　　1　　2　　4　　7
	*/
	void nextPermutation(vector<int>& nums) {
		int i = nums.size() - 2, j = nums.size() - 1;
		while (i >= 0 && nums[i] >= nums[i + 1]) i--;
		if (i >= 0) {
			while (nums[j] <= nums[i]) j--;
			swap(nums[i], nums[j]);
			reverse(nums.begin() + i + 1, nums.end());
			return;
		}
		sort(nums.begin(), nums.end(), less<int>());
	}
};

// 39. 组合总和
class Solution39 {
public:
	// DFS
	void getCombination(vector<int>& candidates, int target, int start, vector<int> &out, vector<vector<int>> &res) {
		if (target < 0) return;
		else if (target == 0) {
			res.push_back(out);
			return;
		}
		int len = candidates.size();
		for (int i = start; i < len; i++) {
			out.push_back(candidates[i]);
			getCombination(candidates, target - candidates[i], i, out, res);
			out.pop_back();
		}
	}
	vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
		vector<vector<int>> res;
		vector<int> out;
		sort(candidates.begin(), candidates.end());
		getCombination(candidates, target, 0, out, res);
		return res;
	}
};

// 40. 组合总和 II
class Solution40 {
public:
	void getCombination(vector<int>& candidates, int target, int start, vector<int> &out, vector<vector<int>> &res) {
		if (target < 0) return;
		else if (target == 0) {
			res.push_back(out);
			return;
		}
		int len = candidates.size();
		for (int i = start; i < len; i++) {
			if (i > start && candidates[i] == candidates[i - 1]) continue;
			out.push_back(candidates[i]);
			getCombination(candidates, target - candidates[i], i + 1, out, res);
			out.pop_back();
		}
	}
	vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
		vector<vector<int>> res;
		vector<int> out;
		sort(candidates.begin(), candidates.end());
		getCombination(candidates, target, 0, out, res);
		return res;
	}
};

// 42. Trapping Rain Water
class Solution42 {
public:
	int trap(vector<int>& height) {
		int left = 0, right = height.size() - 1;
		int minm, res = 0;
		while (left < right) {
			minm = min(height[left], height[right]);
			if (height[left] == minm) {
				left++;
				while (left < right && height[left] < minm) {
					res += minm - height[left++];
				}
			}
			else {
				right--;
				while (left < right && height[right] < minm) {
					res += minm - height[right--];
				}
			}
		}
		return res;
	}
};

// 46. Permutations 全排列
class Solution46 {
public:
	vector<vector<int>> permute(vector<int>& nums) {
		vector<vector<int>> res;
		vector<int> vis(nums.size(), 0);
		vector<int> out;
		permuteDFS(nums, 1, vis, out, res);
		return res;
	}
	void permuteDFS(vector<int>& nums, int level, vector<int>& vis, vector<int>& out, vector<vector<int>>& res) {
		if (level == nums.size() + 1) {
			res.push_back(out);
			return;
		}
		for (int i = 0; i < nums.size(); i++) {
			if (vis[i] == 0) {
				vis[i] = 1;
				out.push_back(nums[i]);
				permuteDFS(nums, level + 1, vis, out, res);
				out.pop_back();
				vis[i] = 0;
			}
		}
	}
};

// 47. Permutations II 全排列II
class Solution47 {
public:
	/*
	是46题的延伸。由于输入数组有可能出现重复数字，如果按照之前的算法运算，会有重复排列产生，我们要避免重复的产生，
	在递归函数中要判断前面一个数和当前的数是否相等，如果相等，前面的数必须已经使用了，
	即对应的visited中的值为1，当前的数字才能使用，否则需要跳过，这样就不会产生重复排列了
	*/
	vector<vector<int>> permuteUnique(vector<int>& nums) {
		vector<vector<int>> res;
		vector<int> vis(nums.size(), 0);
		vector<int> out;
		sort(nums.begin(), nums.end());      // 注意这里也要排个序
		permuteUniqueDFS(nums, 1, vis, out, res);
		return res;
	}
	void permuteUniqueDFS(vector<int>& nums, int level, vector<int>& vis, vector<int>& out, vector<vector<int>>& res) {
		if (level == nums.size() + 1) {
			res.push_back(out);
			return;
		}
		for (int i = 0; i < nums.size(); i++) {
			if (vis[i] == 0) {
				if (i > 0 && nums[i] == nums[i - 1] && vis[i - 1] == 0) continue;
				vis[i] = 1;
				out.push_back(nums[i]);
				permuteUniqueDFS(nums, level + 1, vis, out, res);
				out.pop_back();
				vis[i] = 0;
			}
		}
	}
};

// 56. 合并区间
class Solution56 {
	struct Interval {
		int start;
		int end;
		Interval() : start(0), end(0) {}
		Interval(int s, int e) : start(s), end(e) {}
	};
public:
	vector<Interval> merge(vector<Interval>& intervals) {
		if (intervals.empty()) return {};
		sort(intervals.begin(), intervals.end(), [](Interval a, Interval b) { return a.start < b.start; });
		vector<Interval> res{ intervals[0] };
		int len = intervals.size();
		for (int i = 1; i < len; i++) {
			if (res.back().end < intervals[i].start) res.push_back(intervals[i]);
			else res.back().end = max(res.back().end, intervals[i].end);
		}
		return res;
	}
};

// 76. Minimum Window Substring
// 题意：给定一个原字符串S和一个目标字符串T，让在S中找到一个最短的子串，使得其包含了T中的所有的字母。要求时间复杂度为O(n)
// 解决：哈希表，滑动窗口
class Solution76 {
public:
	string minWindow(string s, string t) {
		string res = "";
		int cnt = 0, left = 0, minLen = INT_MAX;
		unordered_map<char, int> letterCnt;
		for (char letter : t) letterCnt[letter]++;
		for (int i = 0; i < s.size(); i++) {
			letterCnt[s[i]]--;
			if (letterCnt[s[i]] >= 0) cnt++;
			while (cnt == t.size()) {
				if (i - left + 1 < minLen) {
					minLen = i - left + 1;
					res = s.substr(left, minLen);
				}
				letterCnt[s[left]]++;
				if (letterCnt[s[left]] > 0) cnt--;
				left++;
			}
		}
		return res;
	}
};

// 78. 子集
class Solution78 {
public:
	/*我们可以一位一位的网上叠加，比如对于题目中给的例子[1,2,3]来说，最开始是空集，
	那么我们现在要处理1，就在空集上加1，为[1]，现在我们有两个自己[]和[1]，下面我们来处理2，
	我们在之前的子集基础上，每个都加个2，可以分别得到[2]，[1, 2]，那么现在所有的子集合为[], [1], [2], [1, 2]，
	同理处理3的情况可得[3], [1, 3], [2, 3], [1, 2, 3], 再加上之前的子集就是所有的子集合了*/
	// 非递归
	/*vector<vector<int>> subsets(vector<int>& nums) {
		vector<vector<int>> res(1);
		sort(nums.begin(), nums.end());
		int nums_len = nums.size();
		for (int i = 0; i < nums_len; i++) {
			int res_len = res.size();
			for (int j = 0; j < res_len; j++) {
				res.push_back(res[j]);
				res.back().push_back(nums[i]);
			}
		}
		return res;
	}*/
	/*由于原集合每一个数字只有两种状态，要么存在，要么不存在，那么在构造子集时就有选择和不选择两种情况，
	所以可以构造一棵二叉树，	左子树表示选择该层处理的节点，右子树表示不选择，最终的叶节点就是所有子集合，树的结构如下
						 []
					 /          \
					/            \
				   /              \
				[1]                []
			 /       \           /    \
			/         \         /      \
		 [1 2]       [1]       [2]     []
		/     \     /   \     /   \    / \
	[1 2 3] [1 2] [1 3] [1] [2 3] [2] [3] []*/
	//递归
	void getSubsets(vector<int> &nums, int pos, vector<int> &out, vector<vector<int> > &res) {
		res.push_back(out);
		for (int i = pos; i < nums.size(); i++) {
			out.push_back(nums[i]);
			getSubsets(nums, i + 1, out, res);
			out.pop_back();
		}
	}
	vector<vector<int>> subsets(vector<int>& nums) {
		vector<vector<int>> res;
		vector<int> out;
		getSubsets(nums, 0, out, res);
		return res;
	}
};

// 90. 子集 II
class Solution90 {
public:
	/*拿题目中的例子[1 2 2]来分析，根据之前 Subsets 子集合 里的分析可知，当处理到第一个2时，
	此时的子集合为[], [1], [2], [1, 2]，而这时再处理第二个2时，如果在[]和[1]后直接加2会产生重复，
	所以只能在上一个循环生成的后两个子集合后面加2，发现了这一点，题目就可以做了，我们用last来记录上一个处理的数字，
	然后判定当前的数字和上面的是否相同，若不同，则循环还是从0到当前子集的个数，若相同，
	则新子集个数减去之前循环时子集的个数当做起点来循环，这样就不会产生重复了，*/
	// 非递归
	/*vector<vector<int>> subsetsWithDup(vector<int>& nums) {
		vector<vector<int>> res(1);
		sort(nums.begin(), nums.end());
		int len = 1, last = nums[0];
		for (int i = 0; i < nums.size(); i++) {
			if (last != nums[i]) {
				last = nums[i];
				len = res.size();
			}
			int new_len = res.size();
			for (int j = new_len - len; j < new_len; j++) {
				res.push_back(res[j]);
				res.back().push_back(nums[i]);
			}
		}
		return res;
	}*/
	/*在处理到第二个2时，由于前面已经处理了一次2，
	这次我们只在添加过2的[2] 和 [1 2]后面添加2，其他的都不添加，那么这样构成的二叉树如下图所示：
						  []
					 /          \
					/            \
				   /              \
				[1]                []
			 /       \           /    \
			/         \         /      \
		 [1 2]       [1]       [2]     []
		/     \     /   \     /   \    / \
	[1 2 2] [1 2]  X   [1]  [2 2] [2] X  []*/
	// 递归
	void getSubsets(vector<int> &nums, int pos, vector<int> &out, vector<vector<int> > &res) {
		res.push_back(out);
		for (int i = pos; i < nums.size(); i++) {
			out.push_back(nums[i]);
			getSubsets(nums, i + 1, out, res);
			out.pop_back();
			while (i + 1 < nums.size() && nums[i] == nums[i + 1]) i++;
		}
	}
	vector<vector<int>> subsetsWithDup(vector<int>& nums) {
		vector<vector<int>> res;
		vector<int> out;
		getSubsets(nums, 0, out, res);
		return res;
	}
};

// 94. 二叉树的中序遍历
class Solution94 {
	struct TreeNode {
		int val;
		TreeNode *left;
		TreeNode *right;
		TreeNode(int x) : val(x), left(NULL), right(NULL) {}
	};
public:
	/*void inOrder(TreeNode *root, vector<int> &res) {
		if (root) {
			inOrder(root->left, res);
			res.push_back(root->val);
			inOrder(root->right, res);
		}
	}
	vector<int> inorderTraversal(TreeNode* root) {
		vector<int> res;
		inOrder(root, res);
		return res;
	}*/
	vector<int> inorderTraversal(TreeNode* root) {
		vector<int> res;
		stack<TreeNode*> s;
		TreeNode *p = root;
		while (!s.empty() || p) {
			if (p) {
				s.push(p);
				p = p->left;
			}
			else {
				TreeNode *t = s.top(); s.pop();
				res.push_back(t->val);
				p = t->right;
			}
		}
		return res;
	}
};

// 101. Symmetric Tree 镜像二叉树
class Solution101 {
	struct TreeNode {
		int val;
		TreeNode *left;
		TreeNode *right;
		TreeNode(int x) : val(x), left(NULL), right(NULL) {}
	};
public:
	bool isSymmetric(TreeNode* root) {
		if (!root) return true;
		return isSymmetric(root->left, root->right);
	}
	bool isSymmetric(TreeNode* left, TreeNode* right) {
		if (!left && !right) return true;
		if (!left && right || left && !right || left && right && left->val != right->val) return false;
		return isSymmetric(left->left, right->right) && isSymmetric(left->right, right->left);
	}
};

// 126. 单词接龙 II
class Solution126 {
public:
	vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
		unordered_set<string> dict(wordList.begin(), wordList.end());
		vector<vector<string>> ans;
		queue<vector<string>> paths;  // 用path广搜，而不是words
		paths.push({ beginWord });
		int level = 1;
		int minLevel = INT_MAX;
		unordered_set<string> visited;
		while (!paths.empty()) {
			vector<string> path = paths.front();
			paths.pop();
			int path_len = path.size();
			if (path_len > level) {
				for (string w : visited) dict.erase(w);
				visited.clear();
				if (path_len > minLevel)
					break;
				else
					level = path_len;
			}
			string last = path.back();
			int last_len = last.size();
			for (int i = 0; i < last_len; ++i) {
				string news = last;
				for (char c = 'a'; c <= 'z'; ++c) {
					news[i] = c;
					if (dict.count(news)) {
						vector<string> newpath = path;
						newpath.push_back(news);
						visited.insert(news);
						if (news == endWord) {
							minLevel = level;
							ans.push_back(newpath);
						}
						else
							paths.push(newpath);
					}
				}
			}
		}
		return ans;
	}
};

// 127. 单词接龙
class Solution127 {
public:
	/*这里用到了两个高级数据结构unordered_map和queue，即哈希表和队列，
	其中哈希表是记录单词和目前序列长度之间的映射，而队列的作用是保存每一个要展开的单词。
	首先把起始单词映射为1，把起始单词排入队列中，开始队列的循环，取出队首词，然后对其每个位置上的字符，
	用26个字母进行替换，如果此时和结尾单词相同了，就可以返回取出词在哈希表中的值加一。
	如果替换词在字典中存在但在哈希表中不存在，则将替换词排入队列中，并在哈希表中的值映射为之前取出词加一。
	如果循环完成则返回0。*/
	int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
		unordered_set<string> dict(wordList.begin(), wordList.end());  // 将原vector排序后放入unordered_set，方便查找
		unordered_map<string, int> m;  // 每个word对应的转换次数
		queue<string> q;  // 用queue来进行广搜
		m[beginWord] = 1;
		q.push(beginWord);
		while (!q.empty()) {
			string word = q.front();
			q.pop();
			int word_len = word.size();
			for (int i = 0; i < word_len; i++) {
				string newWord = word;
				for (char ch = 'a'; ch <= 'z'; ch++) {
					newWord[i] = ch;
					if (dict.count(newWord)) {
						if (!m.count(newWord)) {
							m[newWord] = m[word] + 1;
							q.push(newWord);
						}
						if (endWord == newWord) return m[newWord];
					}
				}
			}
		}
		return 0;
	}
};

// 128. 最长连续序列
// 在未排序的数组中，找到序列中最长连续数字的长度，要求时间复杂度O(n)
class Solution128 {
public:
	int longestConsecutive(vector<int>& nums) {
		int res = 0;
		unordered_set<int> s(nums.begin(), nums.end());
		for (int num : nums) {
			if (!s.count(num)) continue;
			s.erase(num);
			int pre = num - 1, next = num + 1;
			while (s.count(pre)) s.erase(pre--);
			while (s.count(next)) s.erase(next++);
			res = max(res, next - pre - 1);
		}
		return res;
	}
};

// 141. 环形链表
class Solution141 {
	struct ListNode {
		int val;
		ListNode *next;
		ListNode(int x) : val(x), next(NULL) {}
	};
public:
	// set集合
	/*bool hasCycle(ListNode *head) {
		ListNode *node = head;
		set<ListNode *> nodes_seen;
		while (node != NULL) {
			if (nodes_seen.find(node) != nodes_seen.end()) {
				return true;
			}
			else {
				nodes_seen.insert(node);
			}
			node = node->next;
		}
		return false;
	}*/
	// 双指针，一个跑得快，一个跑得慢
	bool hasCycle(ListNode *head) {
		if (head == NULL || head->next == NULL) return false;
		ListNode *first = head, *second = head->next;
		while (first != second) {
			if (second == NULL || second->next == NULL) return false;
			first = first->next;
			second = second->next->next;
		}
		return true;
	}
};

// 145. Binary Tree Postorder Traversal
// 后序遍历的非递归写法
class Solution145 {
	struct TreeNode {
		int val;
		TreeNode *left;
		TreeNode *right;
		TreeNode(int x) : val(x), left(NULL), right(NULL) {}
	};
public:
	vector<int> postorderTraversal(TreeNode* root) {
		if (!root) return {};
		vector<int> res;
		stack<TreeNode*> s{ {root} };
		TreeNode *head = root;
		while (!s.empty()) {
			TreeNode *t = s.top();
			if ((!t->left && !t->right) || t->left == head || t->right == head) {
				res.push_back(t->val);
				s.pop();
				head = t;
			}
			else {
				if (t->right) s.push(t->right);
				if (t->left) s.push(t->left);
			}
		}
		return res;
	}
};

// 191. 位1的个数
class Solution191 {
public:
	int hammingWeight(uint32_t n) {
		int cnt = 0;
		while (n > 0) {
			cnt += (n & 1);
			n >>= 1;
		}
		return cnt;
	}
};

// 198. 打家劫舍
class Solution198 {
public:
	/*这道题的本质相当于在一列数组中取出一个或多个不相邻数，使其和最大。
	那么我们对于这类求极值的问题首先考虑动态规划Dynamic Programming来解。
	递推公式dp[i] = max(num[i] + dp[i - 2], dp[i - 1]),
	由此看出我们需要初始化dp[0]和dp[1]，其中dp[0]即为num[0]，dp[1]此时应该为max(num[0], num[1])*/
	/*int rob(vector<int>& nums) {
		int len = nums.size();
		if (len <= 1) return nums.empty() ? 0 : nums[0];
		int *dp = new int[len];
		dp[0] = nums[0];
		dp[1] = max(nums[0], nums[1]);
		for (int i = 2; i < len; i++) {
			dp[i] = max(dp[i - 2] + nums[i], dp[i - 1]);
		}
		return dp[len - 1];
	}*/
	int rob(vector<int> &nums) {
		if (nums.size() <= 1) return nums.empty() ? 0 : nums[0];
		vector<int> dp = { nums[0], max(nums[0], nums[1]) };
		int nums_len = nums.size();
		for (int i = 2; i < nums_len; ++i) {
			dp.push_back(max(nums[i] + dp[i - 2], dp[i - 1]));
		}
		return dp.back();
	}
};

// 213. 打家劫舍 II
class Solution213 {
public:
	/*这道题是之前那道House Robber 打家劫舍的拓展，
	现在房子排成了一个圆圈，则如果抢了第一家，就不能抢最后一家，因为首尾相连了，
	所以第一家和最后一家只能抢其中的一家，或者都不抢，那我们这里变通一下，
	如果我们把第一家和最后一家分别去掉，各算一遍能抢的最大值，然后比较两个值取其中较大的一个即为所求。*/
	int rob(vector<int>& nums) {
		if (nums.size() <= 1) return nums.empty() ? 0 : nums[0];
		return max(rob(nums, 0, nums.size() - 1), rob(nums, 1, nums.size()));
	}
	int rob(vector<int> &nums, int left, int right) {
		if (right - left <= 1) return nums[left];
		vector<int> dp(right, 0);
		dp[left] = nums[left];
		dp[left + 1] = max(nums[left], nums[left + 1]);
		for (int i = left + 2; i < right; ++i) {
			dp[i] = max(nums[i] + dp[i - 2], dp[i - 1]);
		}
		return dp[right - 1];
	}
};

// 225. Implement Stack using Queues
/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack obj = new MyStack();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.top();
 * bool param_4 = obj.empty();
 */
class MyStack {
private:
	queue<int> q;
public:
	/** Initialize your data structure here. */
	MyStack() {

	}

	/** Push element x onto stack. */
	void push(int x) {
		queue<int> tmp;
		while (!q.empty()) {
			tmp.push(q.front());
			q.pop();
		}
		q.push(x);
		while (!tmp.empty()) {
			q.push(tmp.front());
			tmp.pop();
		}
	}

	/** Removes the element on top of the stack and returns that element. */
	int pop() {
		int x = q.front();
		q.pop();
		return x;
	}

	/** Get the top element. */
	int top() {
		return q.front();
	}

	/** Returns whether the stack is empty. */
	bool empty() {
		return q.empty();
	}
};

// 231. 2的幂
class Solution231 {
public:
	/*
	1    2     4      8       16 　　....
	1    10    100    1000    10000　....
	很容易看出来2的次方数都只有一个1，剩下的都是0，
	所以我们只要每次判断最低位是否为1，然后向右移位，最后统计1的个数即可判断是否是2的次方数*/
	/*bool isPowerOfTwo(int n) {
		int cnt = 0;
		while (n > 0) {
			cnt += (n & 1);
			n >>= 1;
		}
		return cnt == 1;
	}*/
	/*如果一个数是2的次方数的话，根据上面分析，那么它的二进数必然是最高位为1，其它都为0，
	那么如果此时我们减1的话，则最高位会降一位，其余为0的位现在都为变为1，那么我们把两数相与，就会得到0*/
	bool isPowerOfTwo(int n) {
		return n > 0 && !(n & (n - 1));
	}
};

// 237. Delete Node in a Linked List
class Solution237 {
	struct ListNode {
		int val;
		ListNode *next;
		ListNode(int x) : val(x), next(NULL) {}
	};
public:
	void deleteNode(ListNode* node) {
		node->val = node->next->val;
		ListNode *tmp = node->next;
		node->next = tmp->next;
		delete tmp;    // 注意别忘了释放内存
	}
};

// 326. 3的幂
class Solution326 {
public:
	bool isPowerOfThree(int n) {
		return n > 0 && int(log10(n) / log10(3)) - log10(n) / log10(3) == 0;
	}
};

// 337. 打家劫舍 III
class Solution337 {
	struct TreeNode {
		int val;
		TreeNode *left;
		TreeNode *right;
		TreeNode(int x) : val(x), left(NULL), right(NULL) {}
	};
public:
	/*我们对于某一个节点，如果其左子节点存在，我们通过递归调用函数，算出不包含左子节点返回的值，
	同理，如果右子节点存在，算出不包含右子节点返回的值，那么此节点的最大值可能有两种情况，
	一种是该节点值加上不包含左子节点和右子节点的返回值之和，另一种是左右子节点返回值之和不包含当期节点值，取较大值返回,
	为防止重复计算浪费时间，我们可以把已经算过的节点用哈希表保存起来*/
	int dfs(TreeNode *root, unordered_map<TreeNode *, int> &m) {
		if (!root) return 0;
		if (m.count(root)) return m[root];
		int value1 = dfs(root->left, m) + dfs(root->right, m), value2 = 0;
		if (root->left) value2 += dfs(root->left->left, m) + dfs(root->left->right, m);
		if (root->right) value2 += dfs(root->right->left, m) + dfs(root->right->right, m);
		int value = max(value1, value2 + root->val);
		m[root] = value;
		return value;
	}
	int rob(TreeNode* root) {
		unordered_map<TreeNode *, int> m;
		return dfs(root, m);
	}
};

// 342. 4的幂
class Solution342 {
public:
	/*bool isPowerOfFour(int num) {
		return num > 0 && int(log10(num) / log10(4)) - log10(num) / log10(4) == 0;
	}*/
	/*
	4^0      4^1      4^2      4^3      4^4 　　   ....
	1        100      10000    1000000  100000000  ....
	2^0      2^2      2^4      2^6      2^8   　   ....
	即在二进制中，4次幂可转换为2的偶次幂
	(0x55555555) <==> 01010101010101010101010101010101
	将该数与(0x55555555)做与操作，如果得到的数还是其本身，则可以肯定其为4的次方数*/
	bool isPowerOfFour(int num) {
		return num > 0 && !(num & (num - 1)) && (num & 0x55555555) == num;
	}
};

// 416. 分割等和子集
class Solution416 {
public:
	/*我们定义一个一维的dp数组，其中dp[i]表示数字i是否是原数组的任意个子集合之和，
	那么我们我们最后只需要返回dp[target]就行了。初始化dp[0]为true。
	状态转移方程：我们需要遍历原数组中的数字，对于遍历到的每个数字nums[i]，更新dp数组中[nums[i], target]之间的值，
	对于这个区间中的任意一个数字j，如果dp[j - nums[i]]为true的话，那么dp[j]就一定为true，于是状态转移方程如下：
	dp[j] = dp[j] || dp[j - nums[i]] (nums[i] <= j <= target)*/
	bool canPartition(vector<int>& nums) {
		// 头文件numeric
		int num_sum = accumulate(nums.begin(), nums.end(), 0), target = num_sum >> 1;
		if (num_sum & 1) return false;
		vector<bool> dp(target + 1, false);
		dp[0] = true;
		for (int num : nums) {
			for (int i = target; i >= num; i--) {
				dp[i] = dp[i] || dp[i - num];
			}
		}
		return dp[target];
	}
};

// 445. 两数相加 II
class Solution445 {
	struct ListNode {
		int val;
		ListNode *next;
		ListNode(int x) : val(x), next(NULL) {}
	};
public:
	/*
	我们可以利用栈来保存所有的元素，然后利用栈的后进先出的特点就可以从后往前取数字了，
	我们首先遍历两个链表，将所有数字分别压入两个栈s1和s2中，我们建立一个值为0的res节点，
	然后开始循环，如果栈不为空，则将栈顶数字加入sum中，然后将res节点值赋为sum%10，然后新建一个进位节点head，
	赋值为sum/10，如果没有进位，那么就是0，然后我们head后面连上res，将res指向head，这样循环退出后，
	我们只要看res的值是否为0，为0返回res->next，不为0则返回res即可
	*/
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
		stack<int> s1, s2;
		while (l1) {
			s1.push(l1->val);
			l1 = l1->next;
		}
		while (l2) {
			s2.push(l2->val);
			l2 = l2->next;
		}
		ListNode *res = new ListNode(0);
		int sum = 0;
		while (!s1.empty() || !s2.empty()) {
			if (!s1.empty()) { sum += s1.top(); s1.pop(); }
			if (!s2.empty()) { sum += s2.top(); s2.pop(); }
			res->val = sum % 10;
			ListNode *head = new ListNode(sum / 10);
			head->next = res;
			res = head;
			sum /= 10;
		}
		return res->val == 0 ? res->next : res;
	}
};

// 491. 递增子序列
class Solution491 {
public:
	/*对于重复项的处理，最偷懒的方法是使用set，利用其自动去处重复项的机制，然后最后返回时再转回vector即可。
	由于是找递增序列，所以我们需要对递归函数做一些修改，首先题目中说明了递归序列数字至少两个，
	所以只有当当前子序列个数大于等于2时，才加入结果。然后就是要递增，如果之前的数字大于当前的数字，
	那么跳过这种情况，继续循环，参见代码如下*/
	vector<vector<int>> findSubsequences(vector<int>& nums) {
		set<vector<int>> res;
		vector<int> out;
		helper(nums, 0, out, res);
		return vector<vector<int>>(res.begin(), res.end());
	}
	void helper(vector<int>& nums, int start, vector<int>& out, set<vector<int>>& res) {
		if (out.size() >= 2) res.insert(out);
		for (int i = start; i < nums.size(); ++i) {
			if (!out.empty() && nums[i] < out.back()) continue;
			out.push_back(nums[i]);
			helper(nums, i + 1, out, res);
			out.pop_back();
		}
	}
};

// 496. Next Greater Element I
class Solution496 {
public:
	// 暴力
	/*vector<int> nextGreaterElement(vector<int>& findNums, vector<int>& nums) {
		vector<int> res(findNums.size());
		for (int i = 0; i < findNums.size(); ++i) {
			int j = 0, k = 0;
			for (; j < nums.size(); ++j) {
				if (nums[j] == findNums[i]) break;
			}
			for (k = j + 1; k < nums.size(); ++k) {
				if (nums[k] > nums[j]) {
					res[i] = nums[k];
					break;
				}
			}
			if (k == nums.size()) res[i] = -1;
		}
		return res;
	}*/
	// 用unordered_map存储元素与位置的映射
	vector<int> nextGreaterElement(vector<int>& findNums, vector<int>& nums) {
		vector<int> res(findNums.size());
		unordered_map<int, int> m;
		for (int i = 0; i < nums.size(); ++i) {
			m[nums[i]] = i;
		}
		for (int i = 0; i < findNums.size(); ++i) {
			res[i] = -1;
			int start = m[findNums[i]];
			for (int j = start + 1; j < nums.size(); ++j) {
				if (nums[j] > findNums[i]) {
					res[i] = nums[j];
					break;
				}
			}
		}
		return res;
	}
};

// 503. Next Greater Element II
class Solution503 {
public:
	/*vector<int> nextGreaterElements(vector<int>& nums) {
		int n = nums.size();
		vector<int> res(n, -1);
		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < i + n; j++) {
				if (nums[j % n] > nums[i]) {
					res[i] = nums[j % n];
					break;
				}
			}
		}
		return res;
	}*/
	// 用栈来保存访问过的位置
	vector<int> nextGreaterElements(vector<int>& nums) {
		int n = nums.size();
		vector<int> res(n, -1);
		stack<int> st;
		for (int i = 0; i < 2 * n; ++i) {
			int num = nums[i % n];
			while (!st.empty() && nums[st.top()] < num) {
				res[st.top()] = num;
				st.pop();
			}
			if (i < n) st.push(i);
		}
		return res;
	}
};

// 521. Longest Uncommon Subsequence I  最长非共同子序列之一
class Solution521 {
public:
	int findLUSlength(string a, string b) {
		return a == b ? -1 : max(a.size(), b.size());
	}
};

// 572. Subtree of Another Tree
class Solution572 {
	struct TreeNode {
		int val;
		TreeNode *left;
		TreeNode *right;
		TreeNode(int x) : val(x), left(NULL), right(NULL) {}
	};
public:
	bool isSametree(TreeNode* s, TreeNode* t) {
		if (!s && !t) return true;
		if (!s || !t) return false;
		return isSametree(s->left, t->left) && isSametree(s->right, t->right);
	}
	bool isSubtree(TreeNode* s, TreeNode* t) {
		if (!s) return false;
		if (isSametree(s, t)) return true;
		if (s->val != t->val) return false;
		return isSubtree(s->left, t) || isSubtree(s->right, t);
	}
};

// 632. Smallest Range
// 题意：给定一些数组，都是排好序的，求一个最小的范围，使得这个范围内至少会包括每个数组中的一个数字
// 解答：参见76题
class Solution632 {
public:
	vector<int> smallestRange(vector<vector<int>>& nums) {
		vector<pair<int, int>> vec;
		for (int i = 0; i < nums.size(); i++) {
			for (int num : nums[i]) {
				vec.push_back({ num, i });
			}
		}
		sort(vec.begin(), vec.end());
		int cnt = 0, left = 0, minLength = INT_MAX;
		vector<int> res;
		unordered_map<int, int> m;
		for (int i = 0; i < vec.size(); i++) {
			if (m[vec[i].second]++ == 0) cnt++;
			while (cnt == nums.size() && left <= i) {
				if (vec[i].first - vec[left].first < minLength) {
					minLength = vec[i].first - vec[left].first;
					res = { vec[left].first, vec[i].first };
				}
				if (--m[vec[left].second] == 0) cnt--;
				left++;
			}
		}
		return res;
	}
};

//654. Maximum Binary Tree
class Solution654 {
	struct TreeNode {
		int val;
		TreeNode *left;
		TreeNode *right;
		TreeNode(int x) : val(x), left(NULL), right(NULL) {}
	};
public:
	TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
		if (nums.empty()) return NULL;
		int mx = INT_MIN, mxIndex = 0;
		for (int i = 0; i < nums.size(); i++) {
			if (nums[i] > mx) {
				mx = nums[i];
				mxIndex = i;
			}
		}
		TreeNode *node = new TreeNode(mx);
		vector<int> leftVec = vector<int>(nums.begin(), nums.begin() + mxIndex);
		vector<int> rightVec = vector<int>(nums.begin() + mxIndex + 1, nums.end());
		node->left = constructMaximumBinaryTree(leftVec);
		node->right = constructMaximumBinaryTree(rightVec);
		return node;
	}
};

// 659. Split Array into Consecutive Subsequences 将数组分割成连续子序列
class Solution659 {
public:
	bool isPossible(vector<int>& nums) {
		unordered_map<int, int> freq, need;
		for (int num : nums) {
			freq[num]++;
		}
		for (int num : nums) {
			if (freq[num] == 0) continue;
			else if (need[num] > 0) {
				need[num]--;
				need[num + 1]++;
			}
			else if (freq[num + 1] > 0 && freq[num + 2] > 0) {
				freq[num + 1]--;
				freq[num + 2]--;
				need[num + 3]++;
			}
			else {
				return false;
			}
			freq[num]--;
		}
		return true;
	}
};

// 684. Redundant Connection 
class Solution684 {
public:
	/*解题思路1：DFS  16ms
	每加入一条边，就进行环检测，一旦发现了环，就返回当前边。
	对于无向图，我们还是用邻接表来保存，建立每个结点和其所有邻接点的映射。
	注意：用一个变量pre记录上一次递归的结点，避免死循环。*/
	/*vector<int> findRedundantConnection(vector<vector<int>>& edges) {
		unordered_map<int, unordered_set<int>> mmap;
		for (auto edge : edges) {
			if (hasCircle(edge[0], edge[1], mmap, -1)) return edge;
			mmap[edge[0]].insert(edge[1]);
			mmap[edge[1]].insert(edge[0]);
		}
		return {};
	}
	bool hasCircle(int cur, int target, unordered_map<int, unordered_set<int>>& mmap, int pre) {
		if (mmap[cur].count(target)) return true;
		for (int num : mmap[cur]) {
			if (num == pre) continue;
			if (hasCircle(num, target, mmap, cur)) return true;
		}
		return false;
	}*/

	/*解题思路2：并查集  4ms*/
	vector<int> findRedundantConnection(vector<vector<int>>& edges) {
		vector<int> root(2001, -1);
		for (auto& edge : edges) {
			int x = findRoot(root, edge[0]);
			int y = findRoot(root, edge[1]);
			if (x == y) return edge;
			root[y] = x;
		}
	}
	int findRoot(vector<int> &root, int x) {
		int a = x;
		while (root[x] != -1) {
			x = root[x];
		}
		while (root[a] != -1) {
			int temp = root[a];
			root[a] = x;
			a = temp;
		}
		return x;
	}
};

// 685. Redundant Connection II
class Solution685 {
public:
	vector<int> findRedundantDirectedConnection(vector<vector<int>>& edges) {
		int n = edges.size();
		vector<int> root(n + 1, 0);
		vector<int> first, second;
		for (auto& edge : edges) {
			if (root[edge[1]] == 0) {
				root[edge[1]] = edge[0];
			}
			else {
				first = { root[edge[1]], edge[1] };
				second = edge;
				edge[1] = 0;
			}
		}
		for (int i = 0; i <= n; i++) root[i] = i;
		for (auto& edge : edges) {
			if (edge[1] == 0) continue;
			int x = getRoot(root, edge[0]);
			int y = getRoot(root, edge[1]);
			if (x == y) return first.empty() ? edge : first;
			root[x] = y;
		}
		return second;
	}
	int getRoot(vector<int>& root, int i) {
		while (i != root[i]) i = root[i];
		return i;
	}
};

// 701. Insert into a Binary Search Tree
class Solution701 {
	struct TreeNode {
		int val;
		TreeNode *left;
		TreeNode *right;
		TreeNode(int x) : val(x), left(NULL), right(NULL) {}
	};
public:
	TreeNode* insertIntoBST(TreeNode* root, int val) {
		if (root == NULL) {
			return new TreeNode(val);
		}
		if (val > root->val) root->right = insertIntoBST(root->right, val);
		if (val < root->val) root->left = insertIntoBST(root->left, val);
		return root;
	}
};

// 765. Couples Holding Hands
class Solution765 {
public:
	// 贪心
	int minSwapsCouples(vector<int>& row) {
		int n = row.size();
		int res = 0;
		for (int i = 0; i < n; i += 2) {
			if (row[i + 1] == (row[i] ^ 1)) continue;
			res++;
			for (int j = i + 1; j < n; j++) {
				if (row[j] == (row[i] ^ 1)) {
					row[j] = row[i + 1];
					row[i + 1] = (row[i] ^ 1);
					break;
				}
			}
		}
		return res;
	}
};

// 773. 滑动谜题
class Solution773 {
public:
	int slidingPuzzle(vector<vector<int>>& board) {
		int res = 0;
		set < vector<vector<int>> > visited;
		queue < pair<vector<vector<int>>, vector<int>> > q;
		vector<vector<int>> dirs = { {0, -1}, {-1, 0}, {0, 1}, {1, 0} };
		vector<vector<int>> correct = { {1, 2, 3}, {4, 5, 0} };
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 3; j++) {
				if (board[i][j] == 0) {
					q.push({ board, { i, j } });
					break;
				}
			}
		}
		while (!q.empty()) {
			for (int i = q.size() - 1; i >= 0; i--) {
				auto t = q.front().first;
				auto zero = q.front().second;
				q.pop();
				visited.insert(t);
				if (t == correct) return res;
				for (auto dir : dirs) {
					int x = zero[0] + dir[0], y = zero[1] + dir[1];
					if (x < 0 || x >= 2 || y < 0 || y >= 3) continue;
					vector<vector<int>> cand = t;
					swap(cand[zero[0]][zero[1]], cand[x][y]);
					if (visited.count(cand)) continue;
					q.push({ cand, {x, y} });
				}
			}
			res++;
		}
		return -1;
	}
};

// 801. 使序列递增的最小交换次数
class Solution801 {
public:
	/*swap[i]表示范围[0, i]的子数组同时严格递增且当前位置i需要交换的最小交换次数，
	noSwap[i]表示范围[0, i]的子数组同时严格递增且当前位置i不交换的最小交换次数，两个数组里的值都初始化为n
	由于这道题限制了一定能通过交换生成两个同时严格递增的数组，那么两个数组当前位置和前一个位置之间的关系只有两种，
	一种是不用交换，当前位置的数字已经分别大于前一个位置，另一种是需要交换后，当前位置的数字才能分别大于前一个数字。
	那么我们来分别分析一下，如果当前位置已经分别大于前一位置的数了，那么讲道理我们是不需要再进行交换的，
	但是swap[i]限定了我们必须要交换当前位置i，那么既然当前位置要交换，那么前一个位置i-1也要交换，
	同时交换才能继续保证同时递增，这样我们的swap[i]就可以赋值为swap[i-1] + 1了。而noSwap[i]直接赋值为noSwap[i-1]即可，
	因为不需要交换了。第二种情况是需要交换当前位置，才能保证递增。那么swap[i]正好也是要交换当前位置，而前一个位置不能交换，
	那么即可以用noSwap[i-1] + 1来更新swap[i]，而noSwap[i]是不能交换当前位置，那么我们可以通过交换前一个位置来同样实现递增，
	即可以用swap[i-1]来更新noSwap[i]，当循环结束后，我们在swap[n-1]和noSwap[n-1]中返回较小值即可*/
	int minSwap(vector<int>& A, vector<int>& B) {
		int n = A.size();
		vector<int> swap(n, n), noSwap(n, n);
		swap[0] = 1, noSwap[0] = 0;
		for (int i = 1; i < n; i++) {
			if (A[i] > A[i - 1] && B[i] > B[i - 1]) {
				swap[i] = swap[i - 1] + 1;
				noSwap[i] = noSwap[i - 1];
			}
			if (A[i] > B[i - 1] && B[i] > A[i - 1]) {
				swap[i] = min(swap[i], noSwap[i - 1] + 1);
				noSwap[i] = min(noSwap[i], swap[i - 1]);
			}
		}
		return min(swap[n - 1], noSwap[n - 1]);
	}
};

// 851. 喧闹和富有
class Solution851 {
public:
	// DFS
	/*vector<int> G[510];
	int ans[510];
	void dfs(int x, vector<int>& quiet) {
		if (ans[x] < 0x3f3f3f3f) return;
		int res = quiet[x];
		for (int i = 0; i < G[x].size(); i++) {
			dfs(G[x][i], quiet);
			res = min(res, ans[G[x][i]]);
		}
		ans[x] = res;
	}
	vector<int> loudAndRich(vector<vector<int>>& richer, vector<int>& quiet) {
		int len_quiet = quiet.size();
		int len_richer = richer.size();
		for (int i = 0; i < len_richer; i++) {
			G[richer[i][1]].push_back(richer[i][0]);
		}
		for (int i = 0; i < len_quiet; i++) {
			ans[i] = 0x3f3f3f3f;
		}
		vector<int> res;
		for (int i = 0; i < len_quiet; i++) {
			dfs(i, quiet);
			for (int j = 0; j < len_quiet; j++) {
				if (quiet[j] == ans[i]) {
					res.push_back(j);
					break;
				}
			}
		}
		return res;
	}*/
	// BFS
	vector<int> loudAndRich(vector<vector<int>>& richer, vector<int>& quiet) {
		int len_quiet = quiet.size();
		int len_richer = richer.size();
		vector<set<int>> graph(len_quiet, set<int>());
		for (int i = 0; i < len_richer; i++) {
			graph[richer[i][1]].insert(richer[i][0]);
		}
		queue<int> qu;                                      // 广搜队列，保存比每个节点更富有的节点
		vector<int> res(len_quiet, 0);                      // 记录每个节点的最优（安静）节点
		vector<bool> visited(len_quiet, false);
		for (int i = 0; i < len_quiet; i++) {
			res[i] = i;
			qu.push(i);
			visited[i] = true;
			while (!qu.empty()) {
				int cur = qu.front();
				qu.pop();
				for (auto j : graph[cur]) {
					if (visited[j]) {
						if (quiet[res[j]] < quiet[res[i]]) {
							res[i] = res[j];
						}
					}
					else {
						if (quiet[j] < quiet[res[i]]) {
							res[i] = j;
						}
						qu.push(j);
					}
				}
			}
		}
		return res;
	}
};

// 859. 亲密字符串
class Solution859 {
public:
	bool buddyStrings(string A, string B) {
		int A_len = A.size(), B_len = B.size();
		if (A_len != B_len) return false;
		if (A == B) {
			int hash[26] = { 0 };
			for (int i = 0; i < A_len; i++) {
				hash[A[i] - 'a']++;
			}
			for (auto cnt : hash) {
				if (cnt > 1) {
					return true;
				}
			}
			return false;
		}
		else {
			int first = -1, second = -1;
			for (int i = 0; i < A_len; i++) {
				if (A[i] != B[i]) {
					if (first == -1) first = i;
					else if (second == -1) second = i;
					else return false;
				}
			}
			if (second != -1 && A[first] == B[second] && A[second] == B[first]) {
				return true;
			}
			return false;
		}
	}
};

// 860. 柠檬水找零
class Solution860 {
public:
	bool lemonadeChange(vector<int>& bills) {
		int five = 0, ten = 0;
		for (int bill : bills) {
			if (bill == 5)
				five++;
			else if (bill == 10) {
				if (five == 0) return false;
				five--;
				ten++;
			}
			else {
				if (five > 0 && ten > 0) {
					five--;
					ten--;
				}
				else if (five >= 3) {
					five -= 3;
				}
				else {
					return false;
				}
			}
		}
		return true;
	}
};

// 867. 转置矩阵
class Solution867 {
public:
	vector<vector<int>> transpose(vector<vector<int>>& A) {
		vector<vector<int>> B;
		vector<int> t;
		int A_len = A.size(), A_0_len = A[0].size();
		for (int j = 0; j < A_0_len; j++) {
			t.clear();
			for (int i = 0; i < A_len; i++) {
				t.push_back(A[i][j]);
			}
			B.push_back(t);
		}
		return B;
	}
};

// 868. 二进制间距
class Solution868 {
public:
	int binaryGap(int N) {
		int first = -1, curr = 1;
		int res = INT_MIN;
		while (N) {
			int remainder = N % 2;
			if (remainder > 0) {
				if (first == -1) first = curr;
				else {
					res = max(res, curr - first);
					first = curr;
				}
			}
			N = N / 2;
			curr++;
		}
		if (res == INT_MIN) return 0;
		return res;
	}
};

// 872. 叶子相似的树
class Solution872 {
	struct TreeNode {
		int val;
		TreeNode *left;
		TreeNode *right;
		TreeNode(int x) : val(x), left(NULL), right(NULL) {}
	};
public:
	void preOrder(TreeNode* root, vector<int> &vec) {
		if (root->left == NULL && root->right == NULL) {
			vec.push_back(root->val);
		}
		if (root->left) preOrder(root->left, vec);
		if (root->right) preOrder(root->right, vec);
	}
	bool leafSimilar(TreeNode* root1, TreeNode* root2) {
		vector<int> vec1, vec2;
		preOrder(root1, vec1);
		preOrder(root2, vec2);
		int vec1_len = vec1.size(), vec2_len = vec2.size();
		if (vec1_len != vec2_len) return false;
		for (int i = 0; i < vec1_len; i++) {
			if (vec1[i] != vec2[i]) return false;
		}
		return true;
	}
};

// 873. 最长的斐波那契子序列的长度
class Solution873 {
public:
	// 使用 Set 的暴力法
	/*int lenLongestFibSubseq(vector<int>& A) {
		int A_len = A.size();
		unordered_set<int> S(A.begin(), A.end());
		int ans = 0;
		for (int i = 0; i < A_len; i++) {
			for (int j = i + 1; j < A_len; j++) {
				int x = A[i], y = A[j];
				int length = 2;
				while (S.find(x + y) != S.end()) {
					int z = x + y;
					x = y;
					y = z;
					ans = max(ans, ++length);
				}
			}
		}
		return ans >= 3 ? ans : 0;
	}*/
	// 动态规划
	int lenLongestFibSubseq(vector<int>& A) {
		int A_len = A.size();
		unordered_map<int, int> index;
		for (int i = 0; i < A_len; i++) {
			index[A[i]] = i;
		}
		unordered_map<int, int> longest;
		int ans = 0;
		for (int k = 0; k < A_len; k++) {
			for (int j = 0; j < A_len; j++) {
				int value_i = A[k] - A[j];
				if (value_i < A[j] && index.count(value_i)) {
					int i = index[value_i];
					longest[j * A_len + k] = longest[i * A_len + j] + 1;
					ans = max(ans, longest[j * A_len + k] + 2);
				}
			}
		}
		return ans >= 3 ? ans : 0;
	}
};

// 874. 模拟行走机器人
class Solution874 {
	struct myHash {
		size_t operator()(pair<int, int> __val) const {
			return static_cast<size_t>(__val.first * 30001 + __val.second);
		}
	};
public:
	int robotSim(vector<int>& commands, vector<vector<int>>& obstacles) {
		int dx[] = { 0, 1, 0, -1 };
		int dy[] = { 1, 0, -1, 0 };
		int x = 0, y = 0, di = 0;

		unordered_set<pair<int, int>, myHash> obstacleSet;       // unordered_set存储pair类型时需要自定义hash函数
		for (vector<int> obstacle : obstacles) {
			obstacleSet.insert(make_pair(obstacle[0], obstacle[1]));
		}

		int ans = 0;
		for (int cmd : commands) {
			if (cmd == -2) di = (di + 3) % 4;
			else if (cmd == -1) di = (di + 1) % 4;
			else {
				for (int k = 0; k < cmd; k++) {
					int nx = x + dx[di];
					int ny = y + dy[di];
					if (obstacleSet.find(make_pair(nx, ny)) == obstacleSet.end()) {
						x = nx;
						y = ny;
					}
					ans = max(ans, x * x + y * y);
				}
			}
		}
		return ans;
	}
};

// 875. 爱吃香蕉的珂珂
class Solution875 {
public:
	// 二分法
	bool possible(vector<int>& piles, int H, int speed) {
		int time = 0;
		for (int pile : piles) {
			time += (pile - 1) / speed + 1;    // time += (int)(ceil(pile * 1.0 / speed));
		}
		return time <= H;
	}
	int minEatingSpeed(vector<int>& piles, int H) {
		int low = 1, high = INT_MAX;
		while (low < high) {
			int mid = low + ((high - low) >> 1);
			if (possible(piles, H, mid)) {
				high = mid;
			}
			else {
				low = mid + 1;
			}
		}
		return low;
	}
};

int main()
{
	return 0;
}