import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TanXinSolution {
    public boolean canJump(int[] nums) {
        int len = nums.length;
        int cur = 0;
        while (cur!=len-1&&nums[cur]!=0){
            int tmp = 0;
            int l = nums[cur];
            for (int i = cur+1; i < l; i++) {
                if (nums[cur]>tmp){
                    tmp = nums[i];
                    cur = i;
                }
            }
        }
        return len-1==cur;
    }
    public List<Integer> partitionLabels(String s) {
        char[] charArray = s.toCharArray();
        List<Integer> res = new ArrayList<>();
        Map<Character,Integer> map = new HashMap<>();
        for (int i = 0; i < charArray.length; i++) {
            map.put(charArray[i],i);
        }
        System.out.println(map);
        int right = 0;
        int end = map.get(charArray[0]);
/*        for (char c : charArray) {
            if (end == map.get(c)){
                res.add(end-right+1);
                right = end;
            }
            end =  Math.max(map.get(c),end);
        }*/
        for (int i = 0; i < charArray.length; i++) {
            if (i==end){
                res.add(end - right + 1);
                right =end+1;
            }
            end = Math.max(map.get(charArray[i]),end);
        }
        return res;
    }
}
