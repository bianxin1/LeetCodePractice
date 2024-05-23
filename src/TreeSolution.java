import java.util.*;

public class TreeSolution {
    private class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    /*
        给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。
    */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        dfs(res, root);
        return res;
    }

    private void dfs(List<Integer> res, TreeNode root) {
        if (root == null) {
            return;
        }
        dfs(res, root.left);
        res.add(root.val);
        dfs(res, root.right);
    }

    public List<Integer> inorderTraversal1(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        while (root != null || stack.size() > 0) {
            if (root != null) {
                stack.add(root);
                root = root.left;
                //当前节点为空，说明左边走到头了，从栈中弹出节点并保存
                //然后转向右边节点，继续上面整个过程
            } else {
                TreeNode tmp = stack.pop();
                res.add(tmp.val);
                root = tmp.right;
            }
        }
        return res;
    }

    /*
        给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。
    */
    public TreeNode invertTree(TreeNode root) {
        recur(root);
        return root;
    }

    private void recur(TreeNode root) {
        if (root == null) {
            return;
        }
        recur(root.left);
        recur(root.right);
        TreeNode treeNode = new TreeNode();
        treeNode.left = root.left;
        root.left = root.right;
        root.right = treeNode.left;
    }

    //    给你一个二叉树的根节点 root ， 检查它是否轴对称。
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return dfss(root.left, root.right);
    }

    private boolean dfss(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        if (left.val != right.val) {
            return false;
        }
        return dfss(left.left, right.right) && dfss(left.right, right.left);
    }

    int ans;

    public int diameterOfBinaryTree(TreeNode root) {
        ans = 1;
        depth(root);
        return ans - 1;
    }

    public int depth(TreeNode node) {
        if (node == null) {
            return 0; // 访问到空节点了，返回0
        }
        int L = depth(node.left); // 左儿子为根的子树的深度
        int R = depth(node.right); // 右儿子为根的子树的深度
        ans = Math.max(ans, L + R + 1); // 计算d_node即L+R+1 并更新ans
        return Math.max(L, R) + 1; // 返回该节点为根的子树的深度
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        return re(nums, 0, nums.length - 1);
    }

    private TreeNode re(int[] nums, int left, int right) {
        if (left < right) return null;
        int mid = (left + right) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = re(nums, left, mid - 1);
        root.right = re(nums, mid + 1, right);
        return root;
    }

    public boolean isValidBST(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        List<Integer> res = new ArrayList<>();
        while (root != null || stack.size() > 0) {
            if (root != null) {
                stack.add(root);
                root = root.left;
            } else {
                TreeNode tmp = stack.pop();
                res.add(tmp.val);
                root = tmp.right;
            }
        }
        int tmp = res.get(0);
        for (int i = 1; i < res.size(); i++) {
            if (tmp >= res.get(i)) {
                return false;
            } else {
                tmp = res.get(i);
            }
        }
        return true;
    }

    int res, k;

    public int kthSmallest(TreeNode root, int k) {
        this.k = k;
        kth(root);
        return res;
    }

    private void kth(TreeNode root) {
        if (root == null) return;
        kth(root.left);
        if (k == 0) return;
        if (--k == 0) res = root.val;
        kth(root.right);
    }

    List<Integer> anse = new ArrayList<>();

    public List<Integer> rightSideView(TreeNode root) {
        dfsss(root, 0);
        return anse;
    }

    private void dfsss(TreeNode root, int depth) {
        if (root == null) return;
        if (depth == anse.size()) {
            anse.add(root.val);
        }
        dfsss(root.right, depth + 1);
        dfsss(root.left, depth + 1);
    }

    public void flatten(TreeNode root) {
        TreeNode cur = root;

        while (cur != null) {
            if (cur.left != null) {
                TreeNode tmp = cur.left;
                while (tmp.right != null) {
                    tmp = tmp.right;
                }
                tmp.right = cur.right;
                cur.right = cur.left;
            }
            cur = cur.right;
        }
    }

    int[] preorder;
    HashMap<Integer, Integer> dic = new HashMap<>();

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        this.preorder = preorder;
        for (int i = 0; i < inorder.length; i++) {
            dic.put(inorder[i], i);
        }
        return recur(0, 0, preorder.length - 1);
    }

    private TreeNode recur(int root, int left, int right) {
        if (left > right) return null;
        TreeNode node = new TreeNode(preorder[root]);
        int i = dic.get(preorder[root]);
        node.left = recur(root + 1, left, i - 1);
        node.right = recur(root + i - left + 1, i + 1, right);
        return node;
    }

    Map<Long, Integer> prefixMap;
    int target;

    public int pathSum(TreeNode root, int targetSum) {
        prefixMap = new HashMap<>();
        target = targetSum;

        prefixMap.put(0L, 1);
        return recur(root, 0);

    }

    private int recur(TreeNode root, long curSum) {
        if (root == null) return 0;
        int res = 0;
        curSum += root.val;
        res+=prefixMap.getOrDefault(target-curSum,0);
        prefixMap.put(curSum, prefixMap.getOrDefault(curSum, 0) + 1);
        int left = recur(root.left, curSum);
        int right = recur(root.right, curSum);
        res = res+left+right;
        return res;
    }
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root==null||root==p||root==q) return root;
        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        if (left==null) return right;
        if (right==null) return left;
        return root;
    }
    int maxSum = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        dfssss(root);
        return maxSum;
    }

    private int dfssss(TreeNode root) {
        if (root==null) return 0;
        int left = Math.max(dfssss(root.left),0);
        int right = Math.max(dfssss(root.right),0);
        int cur = left+right+root.val;
        maxSum = Math.max(cur,maxSum);
        return Math.max(left+root.val,right+root.val);
    }

}


