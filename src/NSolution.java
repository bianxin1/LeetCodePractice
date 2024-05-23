import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NSolution {
    char[][] chars;
    List<List<String>> res = new ArrayList<>();

    List<String> cur = new ArrayList<>();
    int n;

    public List<List<String>> solveNQueens(int n) {
        this.n = n;
        chars = new char[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                chars[i][j] = '.';
            }
        }
        dfs(0);
        return res;
    }

    private void dfs(int x) {
        if (x == n) {
            cur.add(Arrays.deepToString(chars));
            res.add(new ArrayList<>(cur));
            return;
        }
        for (int i = x; i < n; i++) {
            if (OK(x, i)) {
                chars[x][i] = 'Q';
                dfs(x + 1);
                chars[x][i] = '.';
            }
        }
    }

    private boolean OK(int x, int i) {
        //纵向查找
        for (int j = 0; j < n; j++) {
            if (chars[j][x] == 'Q'){
                return false;
            }
        }
        //斜方向
        for (int row = x, col = i; row < 0 || col < 0; row--, col--) {
            if (chars[row][col] == 'Q'){
                return false;
            }
        }
        for (int row = x, col = i; row > n || col > n; row++, col++) {
            if (chars[row][col] == 'Q'){
                return false;
            }
        }
        return true;
    }
}
