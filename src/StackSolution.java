import java.util.*;

public class StackSolution {
    private static final Map<Character, Character> map = new HashMap<Character, Character>() {{
        put('{', '}');
        put('[', ']');
        put('(', ')');
        put('?', '?');
    }};

    public boolean isValid(String s) {
        if (s.length() > 0 && !map.containsKey(s.charAt(0))) return false;
        LinkedList<Character> stack = new LinkedList<>() {{
            add('?');
        }};
        for (char c : s.toCharArray()) {
            if (map.containsKey(c)) stack.addLast(c);
            else if (map.get(stack.removeLast()) != c) {
                return false;
            }
        }
        return stack.size() == 1;
    }

    public String decodeString(String s) {
        StringBuilder res = new StringBuilder();
        int multi = 0;
        LinkedList<Integer> stack_multi = new LinkedList<>();
        LinkedList<String> stack_str = new LinkedList<>();
        for (char c : s.toCharArray()) {
            if (c == '[') {
                stack_multi.addLast(multi);
                stack_str.addLast(res.toString());
                multi = 0;
                res = new StringBuilder();
            } else if (c == ']') {
                StringBuilder tmp = new StringBuilder();
                //tmp.append(res.toString()).append(stack_str.removeLast());
                int cur_multi = stack_multi.removeLast();

                for (Integer i = 0; i < cur_multi; i++) {
                    tmp.append(res);
                }
                res = new StringBuilder(stack_str.removeLast() + tmp);
            } else if (c >= '1' && c <= '9') {
                multi = multi * 10 + Integer.parseInt(c + "");
            } else {
                res.append(c);
            }
        }
        return res.toString();
    }

    public int[] dailyTemperatures(int[] temperatures) {
        int[] res = new int[temperatures.length];
        Deque<Integer> deque = new ArrayDeque<>();
        int n = temperatures.length;
        for (int i = n - 1; i >= 0; i--) {
            int t = temperatures[i];
            while (!deque.isEmpty() && t >= temperatures[deque.peek()]) {
                deque.pop();
            }
            if (!deque.isEmpty()) {
                res[i] = deque.peek() - i;
            }
            deque.push(i);
        }
        return res;
    }

    public int largestRectangleArea(int[] heights) {
        int len = heights.length;
        if (len==0) return 0;
        if (len==1) return heights[0];
        int res = 0;
        Deque<Integer> stack = new ArrayDeque<>();
        int[] newHeights = new int[len+2];
        newHeights[0] = 0;
        System.arraycopy(heights, 0, newHeights, 1, len);
        newHeights[len+1] = 0;
        len+=2;
        heights = newHeights;
        stack.push(0);
        for (int i = 0; i <len ; i++) {
            while (heights[i]<heights[stack.peek()]){
                int cur = heights[stack.pop()];
                int width = i - stack.peek()-1;
                res = Math.max(res,cur*width);
            }
            stack.push(i);
        }
        return res;
    }
}
