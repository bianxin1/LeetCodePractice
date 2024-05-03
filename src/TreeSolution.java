import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

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
        if (root==null){
            return;
        }
        recur(root.left);
        recur(root.right);
        TreeNode treeNode = new TreeNode();
        treeNode.left = root.left;
        root.left=root.right;
        root.right = treeNode.left;
    }

}
