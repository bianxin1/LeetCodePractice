import java.util.HashMap;
import java.util.List;

// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
public class Main {
    public static void main(String[] args) {
        String s = "aaaaaaaaaa";
        String p  ="aaaaaaaaaaaaa";
        Solution solution = new Solution();
        int i = solution.lengthOfLongestSubstring(s);
        List<Integer> anagrams = solution.findAnagrams(s, p);
       // System.out.println(i);
        System.out.println(anagrams);
    }
}