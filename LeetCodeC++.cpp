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

// 1. ����֮��
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

// 2. �������
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

// 3. ���ظ��ַ�����Ӵ�
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

// 4. ���������������λ��
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

// 5. ������Ӵ�
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

// 6. Z���α任
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

// 7. ��ת����
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

// 8. �ַ���ת���� (atoi)
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

// 9. ������
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

// 11. ʢ���ˮ������
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

// 19. ɾ������ĵ�����N���ڵ�
class Solution19 {
	struct ListNode {
		int val;
		ListNode *next;
		ListNode(int x) : val(x), next(NULL) {}
	};
public:
	// һ��ָ�룬��������
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
	// ����ָ�룬���n����ͬʱ����������һ��
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

// 21. �ϲ�������������
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
	// �ݹ�
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

// 22. ��������
class Solution22 {
public:
	// ������
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
	// ���ݷ�
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
// ���⣺�ϲ� k �����������
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
	// ��1����Ϊ (k + 1) / 2 �飬�����ϲ�
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

	// ��2�����ȶ���
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

// 26. ɾ�����������е��ظ���
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

// 29. �������
class Solution29 {
public:
	// ���������Ҫ��ʹ�ó˷��������� mod �����
	// ʹ��log��a/b = e^(ln(a))/e^(ln(b)) = e^(ln(a)-ln(b))
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

// 31. Next Permutation ��һ������
class Solution31 {
public:
	/*
	ͨ���۲�ԭ������Է��֣������ĩβ��ǰ���������𽥱�󣬵���2ʱ�ż�С�ģ�
	Ȼ�������ٴӺ���ǰ�ҵ�һ����2������֣���3����ô���ǽ���2��3���ٰѴ�ʱ3�������������ת��һ�¼��ɣ��������£�
	1����2����7����4����3����1
	1����3����7����4����2����1
	1����3����1����2����4����7
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

// 39. ����ܺ�
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

// 40. ����ܺ� II
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

// 46. Permutations ȫ����
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

// 47. Permutations II ȫ����II
class Solution47 {
public:
	/*
	��46������졣�������������п��ܳ����ظ����֣��������֮ǰ���㷨���㣬�����ظ����в���������Ҫ�����ظ��Ĳ�����
	�ڵݹ麯����Ҫ�ж�ǰ��һ�����͵�ǰ�����Ƿ���ȣ������ȣ�ǰ����������Ѿ�ʹ���ˣ�
	����Ӧ��visited�е�ֵΪ1����ǰ�����ֲ���ʹ�ã�������Ҫ�����������Ͳ�������ظ�������
	*/
	vector<vector<int>> permuteUnique(vector<int>& nums) {
		vector<vector<int>> res;
		vector<int> vis(nums.size(), 0);
		vector<int> out;
		sort(nums.begin(), nums.end());      // ע������ҲҪ�Ÿ���
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

// 56. �ϲ�����
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
// ���⣺����һ��ԭ�ַ���S��һ��Ŀ���ַ���T������S���ҵ�һ����̵��Ӵ���ʹ���������T�е����е���ĸ��Ҫ��ʱ�临�Ӷ�ΪO(n)
// �������ϣ����������
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

// 78. �Ӽ�
class Solution78 {
public:
	/*���ǿ���һλһλ�����ϵ��ӣ����������Ŀ�и�������[1,2,3]��˵���ʼ�ǿռ���
	��ô��������Ҫ����1�����ڿռ��ϼ�1��Ϊ[1]�����������������Լ�[]��[1]����������������2��
	������֮ǰ���Ӽ������ϣ�ÿ�����Ӹ�2�����Էֱ�õ�[2]��[1, 2]����ô�������е��Ӽ���Ϊ[], [1], [2], [1, 2]��
	ͬ����3������ɵ�[3], [1, 3], [2, 3], [1, 2, 3], �ټ���֮ǰ���Ӽ��������е��Ӽ�����*/
	// �ǵݹ�
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
	/*����ԭ����ÿһ������ֻ������״̬��Ҫô���ڣ�Ҫô�����ڣ���ô�ڹ����Ӽ�ʱ����ѡ��Ͳ�ѡ�����������
	���Կ��Թ���һ�ö�������	��������ʾѡ��ò㴦��Ľڵ㣬��������ʾ��ѡ�����յ�Ҷ�ڵ���������Ӽ��ϣ����Ľṹ����
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
	//�ݹ�
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

// 90. �Ӽ� II
class Solution90 {
public:
	/*����Ŀ�е�����[1 2 2]������������֮ǰ Subsets �Ӽ��� ��ķ�����֪����������һ��2ʱ��
	��ʱ���Ӽ���Ϊ[], [1], [2], [1, 2]������ʱ�ٴ���ڶ���2ʱ�������[]��[1]��ֱ�Ӽ�2������ظ���
	����ֻ������һ��ѭ�����ɵĺ������Ӽ��Ϻ����2����������һ�㣬��Ŀ�Ϳ������ˣ�������last����¼��һ����������֣�
	Ȼ���ж���ǰ�����ֺ�������Ƿ���ͬ������ͬ����ѭ�����Ǵ�0����ǰ�Ӽ��ĸ���������ͬ��
	�����Ӽ�������ȥ֮ǰѭ��ʱ�Ӽ��ĸ������������ѭ���������Ͳ�������ظ��ˣ�*/
	// �ǵݹ�
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
	/*�ڴ����ڶ���2ʱ������ǰ���Ѿ�������һ��2��
	�������ֻ����ӹ�2��[2] �� [1 2]�������2�������Ķ�����ӣ���ô�������ɵĶ���������ͼ��ʾ��
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
	// �ݹ�
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

// 94. ���������������
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

// 101. Symmetric Tree ���������
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

// 126. ���ʽ��� II
class Solution126 {
public:
	vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
		unordered_set<string> dict(wordList.begin(), wordList.end());
		vector<vector<string>> ans;
		queue<vector<string>> paths;  // ��path���ѣ�������words
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

// 127. ���ʽ���
class Solution127 {
public:
	/*�����õ��������߼����ݽṹunordered_map��queue������ϣ��Ͷ��У�
	���й�ϣ���Ǽ�¼���ʺ�Ŀǰ���г���֮���ӳ�䣬�����е������Ǳ���ÿһ��Ҫչ���ĵ��ʡ�
	���Ȱ���ʼ����ӳ��Ϊ1������ʼ������������У���ʼ���е�ѭ����ȡ�����״ʣ�Ȼ�����ÿ��λ���ϵ��ַ���
	��26����ĸ�����滻�������ʱ�ͽ�β������ͬ�ˣ��Ϳ��Է���ȡ�����ڹ�ϣ���е�ֵ��һ��
	����滻�����ֵ��д��ڵ��ڹ�ϣ���в����ڣ����滻����������У����ڹ�ϣ���е�ֵӳ��Ϊ֮ǰȡ���ʼ�һ��
	���ѭ������򷵻�0��*/
	int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
		unordered_set<string> dict(wordList.begin(), wordList.end());  // ��ԭvector��������unordered_set���������
		unordered_map<string, int> m;  // ÿ��word��Ӧ��ת������
		queue<string> q;  // ��queue�����й���
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

// 128. ���������
// ��δ����������У��ҵ���������������ֵĳ��ȣ�Ҫ��ʱ�临�Ӷ�O(n)
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

// 141. ��������
class Solution141 {
	struct ListNode {
		int val;
		ListNode *next;
		ListNode(int x) : val(x), next(NULL) {}
	};
public:
	// set����
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
	// ˫ָ�룬һ���ܵÿ죬һ���ܵ���
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
// ��������ķǵݹ�д��
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

// 191. λ1�ĸ���
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

// 198. ��ҽ���
class Solution198 {
public:
	/*�����ı����൱����һ��������ȡ��һ����������������ʹ������
	��ô���Ƕ���������ֵ���������ȿ��Ƕ�̬�滮Dynamic Programming���⡣
	���ƹ�ʽdp[i] = max(num[i] + dp[i - 2], dp[i - 1]),
	�ɴ˿���������Ҫ��ʼ��dp[0]��dp[1]������dp[0]��Ϊnum[0]��dp[1]��ʱӦ��Ϊmax(num[0], num[1])*/
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

// 213. ��ҽ��� II
class Solution213 {
public:
	/*�������֮ǰ�ǵ�House Robber ��ҽ������չ��
	���ڷ����ų���һ��ԲȦ����������˵�һ�ң��Ͳ��������һ�ң���Ϊ��β�����ˣ�
	���Ե�һ�Һ����һ��ֻ�������е�һ�ң����߶������������������ͨһ�£�
	������ǰѵ�һ�Һ����һ�ҷֱ�ȥ��������һ�����������ֵ��Ȼ��Ƚ�����ֵȡ���нϴ��һ����Ϊ����*/
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

// 231. 2����
class Solution231 {
public:
	/*
	1    2     4      8       16 ����....
	1    10    100    1000    10000��....
	�����׿�����2�Ĵη�����ֻ��һ��1��ʣ�µĶ���0��
	��������ֻҪÿ���ж����λ�Ƿ�Ϊ1��Ȼ��������λ�����ͳ��1�ĸ��������ж��Ƿ���2�Ĵη���*/
	/*bool isPowerOfTwo(int n) {
		int cnt = 0;
		while (n > 0) {
			cnt += (n & 1);
			n >>= 1;
		}
		return cnt == 1;
	}*/
	/*���һ������2�Ĵη����Ļ������������������ô���Ķ�������Ȼ�����λΪ1��������Ϊ0��
	��ô�����ʱ���Ǽ�1�Ļ��������λ�ήһλ������Ϊ0��λ���ڶ�Ϊ��Ϊ1����ô���ǰ��������룬�ͻ�õ�0*/
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
		delete tmp;    // ע��������ͷ��ڴ�
	}
};

// 326. 3����
class Solution326 {
public:
	bool isPowerOfThree(int n) {
		return n > 0 && int(log10(n) / log10(3)) - log10(n) / log10(3) == 0;
	}
};

// 337. ��ҽ��� III
class Solution337 {
	struct TreeNode {
		int val;
		TreeNode *left;
		TreeNode *right;
		TreeNode(int x) : val(x), left(NULL), right(NULL) {}
	};
public:
	/*���Ƕ���ĳһ���ڵ㣬��������ӽڵ���ڣ�����ͨ���ݹ���ú�����������������ӽڵ㷵�ص�ֵ��
	ͬ��������ӽڵ���ڣ�������������ӽڵ㷵�ص�ֵ����ô�˽ڵ�����ֵ���������������
	һ���Ǹýڵ�ֵ���ϲ��������ӽڵ�����ӽڵ�ķ���ֵ֮�ͣ���һ���������ӽڵ㷵��ֵ֮�Ͳ��������ڽڵ�ֵ��ȡ�ϴ�ֵ����,
	Ϊ��ֹ�ظ������˷�ʱ�䣬���ǿ��԰��Ѿ�����Ľڵ��ù�ϣ��������*/
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

// 342. 4����
class Solution342 {
public:
	/*bool isPowerOfFour(int num) {
		return num > 0 && int(log10(num) / log10(4)) - log10(num) / log10(4) == 0;
	}*/
	/*
	4^0      4^1      4^2      4^3      4^4 ����   ....
	1        100      10000    1000000  100000000  ....
	2^0      2^2      2^4      2^6      2^8   ��   ....
	���ڶ������У�4���ݿ�ת��Ϊ2��ż����
	(0x55555555) <==> 01010101010101010101010101010101
	��������(0x55555555)�������������õ����������䱾������Կ϶���Ϊ4�Ĵη���*/
	bool isPowerOfFour(int num) {
		return num > 0 && !(num & (num - 1)) && (num & 0x55555555) == num;
	}
};

// 416. �ָ�Ⱥ��Ӽ�
class Solution416 {
public:
	/*���Ƕ���һ��һά��dp���飬����dp[i]��ʾ����i�Ƿ���ԭ�����������Ӽ���֮�ͣ�
	��ô�����������ֻ��Ҫ����dp[target]�����ˡ���ʼ��dp[0]Ϊtrue��
	״̬ת�Ʒ��̣�������Ҫ����ԭ�����е����֣����ڱ�������ÿ������nums[i]������dp������[nums[i], target]֮���ֵ��
	������������е�����һ������j�����dp[j - nums[i]]Ϊtrue�Ļ�����ôdp[j]��һ��Ϊtrue������״̬ת�Ʒ������£�
	dp[j] = dp[j] || dp[j - nums[i]] (nums[i] <= j <= target)*/
	bool canPartition(vector<int>& nums) {
		// ͷ�ļ�numeric
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

// 445. ������� II
class Solution445 {
	struct ListNode {
		int val;
		ListNode *next;
		ListNode(int x) : val(x), next(NULL) {}
	};
public:
	/*
	���ǿ�������ջ���������е�Ԫ�أ�Ȼ������ջ�ĺ���ȳ����ص�Ϳ��ԴӺ���ǰȡ�����ˣ�
	�������ȱ��������������������ֱַ�ѹ������ջs1��s2�У����ǽ���һ��ֵΪ0��res�ڵ㣬
	Ȼ��ʼѭ�������ջ��Ϊ�գ���ջ�����ּ���sum�У�Ȼ��res�ڵ�ֵ��Ϊsum%10��Ȼ���½�һ����λ�ڵ�head��
	��ֵΪsum/10�����û�н�λ����ô����0��Ȼ������head��������res����resָ��head������ѭ���˳���
	����ֻҪ��res��ֵ�Ƿ�Ϊ0��Ϊ0����res->next����Ϊ0�򷵻�res����
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

// 491. ����������
class Solution491 {
public:
	/*�����ظ���Ĵ�����͵���ķ�����ʹ��set���������Զ�ȥ���ظ���Ļ��ƣ�Ȼ����󷵻�ʱ��ת��vector���ɡ�
	�������ҵ������У�����������Ҫ�Եݹ麯����һЩ�޸ģ�������Ŀ��˵���˵ݹ�������������������
	����ֻ�е���ǰ�����и������ڵ���2ʱ���ż�������Ȼ�����Ҫ���������֮ǰ�����ִ��ڵ�ǰ�����֣�
	��ô�����������������ѭ�����μ���������*/
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
	// ����
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
	// ��unordered_map�洢Ԫ����λ�õ�ӳ��
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
	// ��ջ��������ʹ���λ��
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

// 521. Longest Uncommon Subsequence I  ��ǹ�ͬ������֮һ
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
// ���⣺����һЩ���飬�����ź���ģ���һ����С�ķ�Χ��ʹ�������Χ�����ٻ����ÿ�������е�һ������
// ��𣺲μ�76��
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

// 659. Split Array into Consecutive Subsequences ������ָ������������
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
	/*����˼·1��DFS  16ms
	ÿ����һ���ߣ��ͽ��л���⣬һ�������˻����ͷ��ص�ǰ�ߡ�
	��������ͼ�����ǻ������ڽӱ������棬����ÿ�������������ڽӵ��ӳ�䡣
	ע�⣺��һ������pre��¼��һ�εݹ�Ľ�㣬������ѭ����*/
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

	/*����˼·2�����鼯  4ms*/
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
	// ̰��
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

// 773. ��������
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

// 801. ʹ���е�������С��������
class Solution801 {
public:
	/*swap[i]��ʾ��Χ[0, i]��������ͬʱ�ϸ�����ҵ�ǰλ��i��Ҫ��������С����������
	noSwap[i]��ʾ��Χ[0, i]��������ͬʱ�ϸ�����ҵ�ǰλ��i����������С���������������������ֵ����ʼ��Ϊn
	���������������һ����ͨ��������������ͬʱ�ϸ���������飬��ô�������鵱ǰλ�ú�ǰһ��λ��֮��Ĺ�ϵֻ�����֣�
	һ���ǲ��ý�������ǰλ�õ������Ѿ��ֱ����ǰһ��λ�ã���һ������Ҫ�����󣬵�ǰλ�õ����ֲ��ֱܷ����ǰһ�����֡�
	��ô�������ֱ����һ�£������ǰλ���Ѿ��ֱ����ǰһλ�õ����ˣ���ô�����������ǲ���Ҫ�ٽ��н����ģ�
	����swap[i]�޶������Ǳ���Ҫ������ǰλ��i����ô��Ȼ��ǰλ��Ҫ��������ôǰһ��λ��i-1ҲҪ������
	ͬʱ�������ܼ�����֤ͬʱ�������������ǵ�swap[i]�Ϳ��Ը�ֵΪswap[i-1] + 1�ˡ���noSwap[i]ֱ�Ӹ�ֵΪnoSwap[i-1]���ɣ�
	��Ϊ����Ҫ�����ˡ��ڶ����������Ҫ������ǰλ�ã����ܱ�֤��������ôswap[i]����Ҳ��Ҫ������ǰλ�ã���ǰһ��λ�ò��ܽ�����
	��ô��������noSwap[i-1] + 1������swap[i]����noSwap[i]�ǲ��ܽ�����ǰλ�ã���ô���ǿ���ͨ������ǰһ��λ����ͬ��ʵ�ֵ�����
	��������swap[i-1]������noSwap[i]����ѭ��������������swap[n-1]��noSwap[n-1]�з��ؽ�Сֵ����*/
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

// 851. ���ֺ͸���
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
		queue<int> qu;                                      // ���Ѷ��У������ÿ���ڵ�����еĽڵ�
		vector<int> res(len_quiet, 0);                      // ��¼ÿ���ڵ�����ţ��������ڵ�
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

// 859. �����ַ���
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

// 860. ����ˮ����
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

// 867. ת�þ���
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

// 868. �����Ƽ��
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

// 872. Ҷ�����Ƶ���
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

// 873. ���쳲����������еĳ���
class Solution873 {
public:
	// ʹ�� Set �ı�����
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
	// ��̬�滮
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

// 874. ģ�����߻�����
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

		unordered_set<pair<int, int>, myHash> obstacleSet;       // unordered_set�洢pair����ʱ��Ҫ�Զ���hash����
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

// 875. �����㽶������
class Solution875 {
public:
	// ���ַ�
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