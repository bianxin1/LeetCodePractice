import java.util.*;

public class Solution {

    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    public int lengthOfLongestSubstring(String s) {
        int len = s.length();
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        int max = 0;
        int left = 0;
        for (int i = 0; i < len; i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i), i);
            max = Math.max(max, i - left + 1);
        }
        return max;
    }

    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> ans = new ArrayList<>();
        int[] sArr = new int[26];
        int[] pArr = new int[26];
        int right = 0;
        int left = 0;
        int slen = s.length();
        int plen = p.length();
        if (slen < plen) {
            return ans;
        }
        for (right = 0; right < p.length(); right++) {
            sArr[s.charAt(right) - 'a']++;
            pArr[p.charAt(right) - 'a']++;
        }
        if (Arrays.equals(sArr, pArr)) {
            ans.add(0);
        }
        for (; right < s.length(); right++) {
            sArr[s.charAt(right) - 'a']++;
            sArr[s.charAt(left) - 'a']--;
            left++;
            if (Arrays.equals(sArr, pArr)) {
                ans.add(left);
            }
        }
        return ans;
    }

    public int subarraySum(int[] nums, int k) {
        int pre = 0;
        int count = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        for (int i = 0; i < nums.length; i++) {
            pre += nums[i];
            if (map.containsKey(pre - k)) {
                count += map.get(pre - k);
            }
            map.put(pre, map.getOrDefault(pre, 0) + 1);

        }
        return count;
    }

    public int maxSubArray(int[] nums) {
        int res = nums[0];
        int dq = 0;
        for (int i = 0; i < nums.length; i++) {
            dq = Math.max(dq + nums[i], nums[i]);
            res = Math.max(dq, res);
        }
        return res;
    }

    /*    以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。*/
    public int[][] merge(int[][] intervals) {
        if (intervals.length == 0) {
            return new int[0][2];
        }
        Arrays.sort(intervals, (v1, v2) -> (v1[0] - v2[0]));
        List<int[]> res = new ArrayList<int[]>();
        for (int i = 0; i < intervals.length; i++) {
            int L = intervals[i][0];
            int R = intervals[i][1];
            if (res.size() == 0 || res.get(res.size() - 1)[1] < L) {
                res.add(new int[]{L, R});
            } else {
                res.get(res.size() - 1)[1] = Math.max(R, res.get(res.size() - 1)[1]);
            }
        }
        return res.toArray(new int[res.size()][2]);
    }

    /*  给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。*/
    public void rotate(int[] nums, int k) {
        k = k % nums.length;
        reserve(nums, 0, nums.length - 1);
        reserve(nums, 0, k - 1);
        reserve(nums, k, nums.length - 1);
    }

    public void reserve(int[] nums, int l, int r) {
        while (l < r) {
            int temp = nums[l];
            nums[l] = nums[r];
            nums[r] = temp;
            l++;
            r--;
        }
    }
/*    给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。

    题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。

    请 不要使用除法，且在 O(n) 时间复杂度内完成此题。*/

    /**
     * 将ans输出数组用作暂存前缀积，在循环动态乘上后缀积以降低空间复杂度
     */
    public int[] productExceptSelf(int[] nums) {
        if (nums.length == 0) {
            return new int[0];
        }
        //qrr存储前缀积，prr存储后缀积
        int[] qrr = new int[nums.length];
        int[] prr = new int[nums.length];
        int[] ans = new int[nums.length];
        int len = nums.length;
        qrr[0] = 1;
        prr[len - 1] = 1;
        //循环求解前缀积以及后缀积
        for (int i = 1; i < len; i++) {
            qrr[i] = qrr[i - 1] * nums[i - 1];
            prr[len - 1 - i] = prr[len - i] * nums[len - i];

        }
        //循环求解
        for (int i = 0; i < len; i++) {
            ans[i] = qrr[i] * prr[i];
        }
        return ans;
    }

    /*    给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。

        请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。*/
    public int firstMissingPositive(int[] nums) {
        //将数组中小于等于0的数更新为N+1
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] <= 0) nums[i] = nums.length + 1;
        }
        //对每一个绝对值在1-N的数，其对应的下标减一置负
        for (int i = 0; i < nums.length; ++i) {
            int num = Math.abs(nums[i]);
            if (num <= nums.length) {
                nums[num - 1] = -Math.abs(nums[num - 1]);
            }
        }
        //在遍历完成之后，如果数组中的每一个数都是负数，那么答案是 N+1，否则答案是第一个正数的位置加 1
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) return i + 1;

        }
        return nums.length + 1;
    }

    /*    给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。*/
    public void setZeroes(int[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;
        boolean row_0 = false;
        boolean col_0 = false;
        //第一行是否有0
        for (int i = 0; i < col; i++) {
            if (matrix[0][i] == 0) {
                row_0 = true;
                break;
            }
        }
        //第一列是否有0
        for (int i = 0; i < row; i++) {
            if (matrix[i][0] == 0) {
                col_0 = true;
                break;
            }
        }
        //一第一行和第一列作为标记
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = matrix[0][j] = 0;
                }
            }
        }
        //依据标记置0
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (row_0) {
            for (int i = 0; i < col; i++) {
                matrix[0][i] = 0;
            }
        }
        if (col_0) {
            for (int i = 0; i < row; i++) {
                matrix[i][0] = 0;
            }
        }
    }

    /*给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。*/
    public List<Integer> spiralOrder(int[][] matrix) {
        if (matrix.length == 0) {
            return new ArrayList<Integer>();
        }
        int l = 0, r = matrix[0].length - 1, t = 0, b = matrix.length - 1, x = 0;
        Integer[] res = new Integer[(r + 1) * (b + 1)];
        while (true) {
            //左到右
            for (int i = l; i <= r; i++) {
                res[x++] = matrix[t][i];
            }
            if (++t > b) break;
            //上到下
            for (int i = t; i <= b; i++) {
                res[x++] = matrix[i][r];
            }
            if (--r < l) break;
            //右到左
            for (int i = r; i >= l; i--) {
                res[x++] = matrix[b][i];
            }
            if (--b < t) break;
            //下到上
            for (int i = b; i >= t; i--) {
                res[x++] = matrix[i][l];
            }
            if (++l > r) break;

        }
        return Arrays.asList(res);
    }

    /*    给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

        你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。*/
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n / 2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][n - j - 1];
                matrix[i][n - j - 1] = temp;
            }
        }
    }

    /*编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

    每行的元素从左到右升序排列。
    每列的元素从上到下升序排列。*/
    public boolean searchMatrix(int[][] matrix, int target) {
        for (int i = 0, j = matrix[0].length - 1; i < matrix.length && j >= 0; ) {
            if (matrix[i][j] < target) i++;
            else if (matrix[i][j] > target) {
                j--;
            } else {
                return true;
            }
        }
        return false;
    }

    /*给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

    返回 滑动窗口中的最大值*/
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums.length == 0 || k == 0) {
            return new int[0];
        }
        Deque<Integer> deque = new LinkedList<>();
        int[] res = new int[nums.length - k + 1];
        for (int i = 0; i < k; i++) {
            while (!deque.isEmpty() && deque.peekLast() < nums[i]) {
                deque.removeLast();
            }
            deque.addLast(nums[i]);
        }
        res[0] = deque.peekFirst();
        for (int i = k; i < nums.length; i++) {
            if (deque.peekFirst() == nums[i - k])
                deque.removeFirst();
            while (!deque.isEmpty() && deque.peekLast() < nums[i]) {
                deque.removeLast();
            }
            deque.addLast(nums[i]);
            res[i - k + 1] = deque.peekFirst();
        }
        return res;
    }

    /*    给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。*/
    public ListNode reverseListDieDai(ListNode head) {
        ListNode cur = head, pre = null;
        while (cur != null) {
            ListNode tmp = cur.next;// 暂存后继节点 cur.next
            cur.next = pre;          // 修改 next 引用指向
            pre = cur;               // pre 暂存 cur
            cur = tmp;               // cur 访问下一节点
        }
        return pre;
    }

    public ListNode reverseListDiGui(ListNode head) {
        return recur(head, null);
    }

    private ListNode recur(ListNode nex, ListNode pre) {
        if (nex == null) return pre;
        ListNode res = recur(nex.next, nex);
        nex.next = pre;
        return res;
    }

    /*给你一个单链表的头节点 head ，请你判断该链表是否为 回文链表。如果是，返回 true ；否则，返回 false 。*/
    public boolean isPalindrome(ListNode head) {
        if (head == null) return true;
        //找到中间节点
        ListNode firstHalfEndNode = Half(head);
        //反转后半链表
        ListNode secondHalfBeginNode = reverseListDieDai(firstHalfEndNode.next);
        ListNode p1 = head;
        ListNode p2 = secondHalfBeginNode;
        Boolean result = true;
        while (result && p2 != null) {
            if (p1.val != p2.val) {
                result = false;
            }
            p1 = p1.next;
            p2 = p2.next;
        }
        reverseListDieDai(firstHalfEndNode.next);
        return result;
    }

    private ListNode Half(ListNode head) {
        ListNode low = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            low = low.next;
            fast = fast.next.next;
        }
        return low;
    }

    /*给你一个链表的头节点 head ，判断链表中是否有环。

    如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。

    如果链表中存在环 ，则返回 true 。 否则，返回 false 。*/
    public boolean hasCycle(ListNode head) {
        if (head == null) {
            return false;
        }
        ListNode low = head;
        ListNode fast = head;
        while (low.next != null && fast.next != null && fast.next.next != null) {
            low = low.next;
            fast = fast.next.next;
            if (low == fast) {
                return true;
            }
        }
        return false;
    }

    /*给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

    如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

    不允许修改 链表。*/

    /**
     * 解法一哈希表
     *
     * @param head
     * @return
     */
    public ListNode detectCycle(ListNode head) {
        if (head == null) {
            return null;
        }
        HashMap<ListNode, Integer> map = new HashMap<ListNode, Integer>();
        Integer count = 0;
        while (head.next != null) {

            if (map.containsKey(head)) {
                return head;
            }
            map.put(head, count);
            head = head.next;
        }
        return null;
    }

    public ListNode detectCycle2(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode low = head;
        ListNode fast = head;
        while (true) {
            if (fast.next == null || fast.next.next == null) {
                return null;
            }
            fast = fast.next.next;
            low = low.next;
            if (low == fast) {
                break;
            }
        }
        fast = head;
        while (low != fast) {
            low = low.next;
            fast = fast.next;
        }
        return fast;
    }

    /*给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

    请你将两个数相加，并以相同形式返回一个表示和的链表。

    你可以假设除了数字 0 之外，这两个数都不会以 0 开头。*/
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode pre = new ListNode(0);
        ListNode cur = pre;
        int crray = 0;
        while (l1 != null || l2 != null) {
            int x = l1 == null ? 0 : l1.val;
            int y = l2 == null ? 0 : l2.val;
            int sum = x + y + crray;
            crray = sum / 10;
            cur.next = new ListNode(sum % 10);
            cur = cur.next;
            if (l1 != null) l1 = l1.next;
            if (l2 != null) l2 = l2.next;
        }
        if (crray == 1) cur.next = new ListNode(crray);
        return pre.next;
    }

    /*给你一个字符串 s，找到 s 中最长的回文 子串。如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。*/
    public String longestPalindrome(String s) {
        //单个字符直接返回
        if (s == null || s.length() < 2) {
            return s;
        }
        //设定初值
        int length = s.length();
        int maxLen = 1;
        int begin = 0;
        Boolean[][] dp = new Boolean[length][length];
        char[] charArray = s.toCharArray();
        //单个字符一定回文
        for (int i = 0; i < length; i++) {
            dp[i][i] = true;
        }
        //右边循环在外
        for (int j = 1; j < length; j++) {
            for (int i = 0; i < j; i++) {
                if (charArray[i] != charArray[j]) {
                    dp[i][j] = false;
                } else {
                    //长度为2或3设置为true
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];//状态转移方程dp[i][j]=dp[i+1][j-1]and s[i]==s[j]
                    }
                }
                //若为回文串且长度大于当前最大值则更新最大值
                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }

            }
        }
        return s.substring(begin, begin + maxLen);//前闭后不闭
    }

    // 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        if (list1 == null) {
            return list2;
        } else if (list2 == null) {
            return list1;
        } else if (list1.val <= list2.val) {
            list1.next = mergeTwoLists(list1.next, list2);
            return list1;
        } else {
            list2.next = mergeTwoLists(list1, list2.next);
            return list2;
        }
    }

    //    给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null) {
            return head;
        }
        ListNode slow = head;
        ListNode fast = head;
        for (int i = 0; i < n; i++) {
            fast = fast.next;
        }
        if (fast == null) return head.next;
        while (fast.next != null) {
            slow = slow.next;
            fast = fast.next;
        }
        slow.next = slow.next.next;
        return head;
    }

    //    给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
    public ListNode swapPairs(ListNode head) {
        if (head == null) {
            return head;
        }
        if (head.next == null) {
            return head;
        }
        ListNode t = swapPairs(head.next.next);
        ListNode l1 = head;
        ListNode l2 = head.next;
        l1.next = t;
        l2.next = l1;
        return l2;

    }

    /*给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
    你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换*/
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode tmp = head;
        ListNode p = null;
        for (int i = 0; i < k - 1; i++) {
            if (head.next == null) {
                return tmp;
            }
            head = head.next;
        }
        ListNode cur = head;
        if (head.next == null) {
            p = reverseListDiGui2(tmp, p);
            return p;
        } else {
            head = head.next;
            p = reverseKGroup(head, k);
            cur.next = p;
            p = reverseListDiGui2(tmp, p);
            return p;
        }
    }

    private ListNode reverseListDiGui2(ListNode head, ListNode p) {
        ListNode res = recur2(head, null, p);
        head.next = p;
        return res;

    }

    private ListNode recur2(ListNode nex, ListNode pre, ListNode p) {
        if (nex == p) return pre;
        ListNode res = recur2(nex.next, nex, p);
        nex.next = pre;
        return res;
    }

    /*
        给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。
    */
    class Node {
        int val;
        Node next;
        Node random;

        public Node(int val) {
            this.val = val;
            this.next = null;
            this.random = null;
        }
    }

    public Node copyRandomList(Node head) {
        if (head == null) {
            return head;
        }
        HashMap<Node, Node> map = new HashMap<>();
        Node cur = head;
        while (cur != null) {
            map.put(cur, new Node(cur.val));
            cur = cur.next;
        }
        cur = head;
        while (cur != null) {
            map.get(cur).next = map.get(cur.next);
            map.get(cur).random = map.get(cur.random);
            cur = cur.next;
        }
        return map.get(head);
    }

    /*给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。*/
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        int length = 0;
        ListNode node = head;
        while (node != null) {
            node = node.next;
            length++;
        }
        //存储结果
        ListNode dummy = new ListNode(0, head);
        for (int intv = 1; intv < length; intv <<= 1) {
            ListNode prev = dummy;
            ListNode cur = dummy.next;
            while (cur != null) {
                ListNode head_1 = cur;
                for (int i = 1; i < intv && cur != null && cur.next != null; i++) {
                    cur = cur.next;

                }
                //分割第一段
                ListNode head_2 = cur.next;
                cur.next = null;
                cur = head_2;
                for (int i = 1; i < intv && cur != null && cur.next != null; i++) {
                    cur = cur.next;
                }
                //第二个连边可能为空
                ListNode next = null;
                if (cur != null) {
                    next = cur.next;
                    cur.next = null;
                }
                ListNode merge = mergeTwoLists(head_1, head_2);
                prev.next = merge;
                while (prev.next != null) {
                    prev = prev.next;
                }
                cur = next;
            }
        }
        return dummy.next;
    }

    /*
        给你一个链表数组，每个链表都已经按升序排列。请你将所有链表合并到一个升序链表中，返回合并后的链表。
    */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists==null||lists.length==0){
            return null;
        }
        PriorityQueue<ListNode> queue = new PriorityQueue(new Comparator<ListNode>() {
            public int compare(ListNode o1, ListNode o2) {
                return (o1.val - o2.val);
            }
        });
        for(int i=0;i<lists.length;i++) {
            while(lists[i] != null) {
                queue.add(lists[i]);
                lists[i] = lists[i].next;
            }
        }
        ListNode dummy = new ListNode(-1);
        ListNode head = dummy;
        //从堆中不断取出元素，并将取出的元素串联起来
        while( !queue.isEmpty() ) {
            dummy.next = queue.poll();
            dummy = dummy.next;
        }
        dummy.next = null;
        return head.next;
    }

}


