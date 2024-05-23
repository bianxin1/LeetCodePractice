import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class MapSolution {
    public int numIslands(char[][] grid) {
        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    dfs(grid, i, j);
                }
            }
        }
        return count;
    }

    private void dfs(char[][] grid, int i, int j) {
        if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] == '0') return;
        grid[i][j] = '0';
        dfs(grid, i + 1, j);
        dfs(grid, i - 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i, j - 1);
    }

    public int orangesRotting(int[][] grid) {
        int rows = grid.length;
        int cols = grid[0].length;
        Queue<int[]> queue = new LinkedList<>();
        int count = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 1) {
                    count++;
                } else if (grid[i][j] == 2) {
                    queue.add(new int[]{i, j});
                }
            }
        }
        int round = 0;
        while (count > 0 && !queue.isEmpty()) {
            round++;
            int n = queue.size();
            for (int k = 0; k < n; k++) {
                int[] orange = queue.poll();
                int i = orange[0];
                int j = orange[1];
                if (i - 1 >= 0 && grid[i - 1][j] == 1) {
                    grid[i - 1][j] = 2;
                    count--;
                    queue.add(new int[]{i - 1, j});
                }
                if (i + 1 < rows && grid[i + 1][j] == 1) {
                    grid[i + 1][j] = 2;
                    count--;
                    queue.add(new int[]{i + 1, j});
                }
                if (j - 1 >= 0 && grid[i][j - 1] == 1) {
                    grid[i][j - 1] = 2;
                    count--;
                    queue.add(new int[]{i, j - 1});
                }
                if (j + 1 < cols && grid[i][j + 1] == 1) {
                    grid[i][j + 1] = 2;
                    count--;
                    queue.add(new int[]{i, j + 1});
                }
            }

        }
        if (count>0){
            return -1;
        }else {
            return round;
        }
    }
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int[] indegrees = new int[numCourses];
        List<List<Integer>> list = new ArrayList<>();
        Queue<Integer> queue = new LinkedList<>();
        //邻接表初始化
        for (int i = 0; i < numCourses; i++) {
            list.add(new ArrayList<>());
        }
        //入度表，邻接表初始化
        for (int[] pq : prerequisites) {
            indegrees[pq[0]]++;
            list.get(pq[1]).add(pq[0]);
        }
        for (int i = 0; i < numCourses; i++) {
            if (indegrees[i]==0){
                queue.add(i);
            }
        }
        while (!queue.isEmpty()){
            int pre = queue.poll();
            numCourses--;
            for (Integer l : list.get(pre)) {
                indegrees[l]--;
                if (indegrees[l]==0){
                    queue.add(l);
                }
            }
        }
        return numCourses==0;
    }
    public boolean canFinishSolution2(int numCourses, int[][] prerequisites){
        List<List<Integer>> list = new ArrayList<>();
        int[] flags = new int[numCourses];
        for (int i = 0; i < numCourses; i++) {
            list.add(new ArrayList<>());
        }
        for (int[] pq : prerequisites) {
            list.get(pq[1]).add(pq[0]);
        }
        for(int i = 0; i < numCourses; i++)
            if(!dfs(list, flags, i)) return false;
        return true;
    }

    private boolean dfs(List<List<Integer>> list, int[] flags, int i) {
        if (flags[i]==1) return false;
        if (flags[i]==-1) return true;
        for (Integer l : list.get(i)) {
            if (!dfs(list,flags,l)) return false;

        }
        flags[i] = -1;
        return true;
    }
}
