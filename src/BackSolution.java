import java.util.*;

public class BackSolution {
    List<Integer> nums;
    List<List<Integer>> res;
    public List<List<Integer>> permute(int[] nums) {
        this.nums = new ArrayList<>();
        this.res = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            this.nums.add(nums[i]);
        }
        dfs(0);
        return res;
    }

    private void dfs(int x) {
        if (x== nums.size()-1){
            res.add(new ArrayList<>(nums));
            return;
        }
        for (int i = x; i < nums.size(); i++) {
            swap(i,x);
            dfs(x+1);
            swap(x,i);
        }
    }

    private void swap(int a, int b) {
        int tmp = nums.get(a);
        nums.set(a, nums.get(b));
        nums.set(b, tmp);
    }
    List<List<Integer>> ans = new ArrayList<>();
    List<Integer> path = new ArrayList<>();
    int num[];
    public List<List<Integer>> subsets(int[] nums) {
        this.num = nums;
        dfss(0);
        return ans;
    }

    private void dfss(int x) {
        if (x==num.length){
            ans.add(new ArrayList<>(path));
            return;
        }
        //不选此数
        dfss(x+1);
        //选此数
        path.add(num[x]);
        dfss(x+1);
        //恢复现场
        path.remove(path.size()-1);
    }
    public List<List<Integer>> subsetsSolution2(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < (1<< nums.length); i++) {
            List<Integer> path = new ArrayList<>();
            for (int j = 0; j < nums.length; j++) {
                if (((i>>j)&1)==1) path.add(num[j]);
            }
            res.add(new ArrayList<>(path));
        }
        return res;
    }
    Map<Character, char[]> nineKeyMap = new HashMap<>();

    List<String> list = new ArrayList<>();
    String digits;
    List<Character> an = new ArrayList<>();
    public List<String> letterCombinations(String digits) {


        // 初始化九键拼音字母对应关系
        nineKeyMap.put('2', new char[]{'a', 'b', 'c'});
        nineKeyMap.put('3', new char[]{'d', 'e', 'f'});
        nineKeyMap.put('4', new char[]{'g', 'h', 'i'});
        nineKeyMap.put('5', new char[]{'j', 'k', 'l'});
        nineKeyMap.put('6', new char[]{'m', 'n', 'o'});
        nineKeyMap.put('7', new char[]{'p', 'q', 'r', 's'});
        nineKeyMap.put('8', new char[]{'t', 'u', 'v'});
        nineKeyMap.put('9', new char[]{'w', 'x', 'y', 'z'});
        System.out.println(nineKeyMap);
        this.digits = digits;
        //dfsss(0);
        return list;
    }

    private void dfsss(int x) {
        if (x==digits.length()){
            String tmp = an.toString();
            list.add(tmp);
            return;
        }
        char c = digits.charAt(x);
        char[] characters = nineKeyMap.get(c);
        an.add(characters[0]);
        dfsss(x+1);
        an.add(characters[1]);
        dfsss(x+1);
        an.add(characters[2]);
        dfsss(x+1);
        an.remove(an.size()-1);
    }
    int target;
    int[] candidates;
    List<List<Integer>> lists = new ArrayList<>();
    List<Integer> tmp = new ArrayList<>();
    int cur = 0;
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        this.target = target;
        this.candidates = candidates;
        dfssss(0);
        return lists;
    }

    private void dfssss(int x) {
        if (x== candidates.length||cur>target){
            return;
        }else if (cur==target){
            lists.add(new ArrayList<>(tmp));
        }
        //不选
        dfssss(x+1);
        cur+=candidates[x];
        tmp.add(candidates[x]);
        //再选？
        dfssss(x);
        cur-=candidates[x];
        tmp.remove(tmp.size()-1);
    }
    List<String> anser = new ArrayList<>();
    StringBuilder str = new StringBuilder();
    public List<String> generateParenthesis(int n) {
        gene(n,0,0);
        return anser;
    }

    private void gene(int n, int open,int close) {
        if (str.length()==n*2){
            anser.add(str.toString());
            return;
        }
        if (open<n){
            str.append('(');
            gene(n,open+1,0);
            str.deleteCharAt(str.length()-1);
        }
        if (close<open){
            str.append(')');
            gene(n,0,close+1);
            str.deleteCharAt(str.length()-1);
        }
    }
    char[][] board;
    String word;
    char[] chars;
    Boolean flag = false;

    public boolean exist(char[][] board, String word) {
        this.board = board;
        this.word = word;
        chars = new char[word.length()];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (word.charAt(0)==board[i][j]){
                    dfs(i,j,0);
                    if (flag){
                        return flag;
                    }
                }
            }
        }
        return flag;
    }

    private void dfs(int i, int j, int x) {
        if (i<0||i>board.length||j<0||j>board[0].length){
            return;
        }
        if (x==word.length()){
            flag = true;
            return;
        }
        if (word.charAt(x) == board[i][j]){
            dfs(i-1,j,x+1);
            dfs(i+1,j,x+1);
            dfs(i,j-1,x+1);
            dfs(i,j+1,x+1);
        }else {
            return;
        }
    }
    int len;
    List<List<String>> listList = new ArrayList<>();
    Deque<String> stack = new ArrayDeque<>();

    public List<List<String>> partition(String s) {
        len = s.length();
        char[] charArray = s.toCharArray();
        dfs(charArray, 0);
        return listList;
    }

    private void dfs(char[] charArray, int x) {
        if (x==charArray.length){
            listList.add(new ArrayList<>(stack));
            return;
        }
        for (int i = x; i <len; i++) {
            if (!isch(charArray,x,i)){
                continue;
            }
            stack.addLast(new String(charArray,x,i+1-x));
            dfs(charArray,i+1);
            stack.removeLast();
        }
    }

    private boolean isch(char[] charArray,int left, int right) {
        while (left < right) {
            if (charArray[left] != charArray[right]) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }

}
