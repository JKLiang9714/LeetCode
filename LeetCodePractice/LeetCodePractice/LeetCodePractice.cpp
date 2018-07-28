// LeetCodePractice.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"

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
				max_len = max_len >(j - i) ? max_len : (j - i);
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
		if (a_start >= a.size()) return b[b_start + k - 1];
		if (b_start >= b.size()) return a[a_start + k - 1];
		if (k == 1) return a[a_start] < b[b_start] ? a[a_start] : b[b_start];
		int a_min = INT_MAX, b_min = INT_MAX;
		if (a_start + k / 2 - 1 < a.size()) a_min = a[a_start + k / 2 - 1];
		if (b_start + k / 2 - 1 < b.size()) b_min = b[b_start + k / 2 - 1];
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


int main()
{
    return 0;
}

