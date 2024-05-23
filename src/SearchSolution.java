public class SearchSolution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int high = 0, low = matrix[0].length;
        int mid;
        int left = 0, right = matrix.length;
        while (high <= low) {
            mid = (matrix[high][right] + matrix[low][right]) / 2;
            if (matrix[mid][right] == target) {
                return true;
            }
            if (matrix[mid][right] < target) {
                high = mid + 1;
            } else {
                low = mid - 1;
            }
        }
        while (left <= right) {
            mid = (matrix[high][left] + matrix[high][right]) / 2;
            if (matrix[high][mid] == target) {
                return true;
            }
            if (matrix[high][mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return false;
    }

    public int[] searchRange(int[] nums, int target) {
        int start = BiSearch(nums, target);
        if (start == nums.length || nums[start] != target) {
            return new int[]{-1, -1}; // nums 中没有 target
        }
        int end = BiSearch(nums, target + 1);
        return new int[]{start, end};

    }

    private int BiSearch(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        int mid;
        while (left <= right) {
            mid = (left + right) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;

    }

    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1, mid;
        if (nums.length == 0) {
            return -1;
        }
        while (left <= right) {
            mid = (left + right) / 2;
            if (nums[mid] == target) {
                return mid;
            }
            if (nums[left] <= nums[mid]) {
                if (target >= nums[left] && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }

            } else {
                if (target <= nums[right] && target > nums[mid]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }

        }
        return -1;
    }
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int m = nums2.length;
        int left = (m+n+1)/2;
        int right = (m+n+2)/2;
        return (recur(nums1,0,n-1,nums2,0,m-1,left)+recur(nums1,0,n-1,nums2,0,m-1,left))/2;
    }

    private double recur(int[] nums1, int start1, int end1, int[] nums2, int start2, int end2, int k) {
        int len1 = end1 - start1 + 1;
        int len2 = end2 - start2 + 1;
        if (len1>len2){
            recur(nums2,start2,end2,nums1,start1,end1,k);
        }
        if (len1==0){
            return nums2[start2+k-1];
        }
        if (k==1){
            return Math.min(nums1[start1],nums2[start2] );
        }
        int i = start1+Math.min(len1,k/2) -1 ;
        int j = start2+Math.min(len2,k/2) -1 ;
        if (nums1[i]<nums2[j]){
            return recur(nums1,i+1,end1,nums2,start2,end2,k-(i-start1+1));
        }else {
            return recur(nums1,start1,end1,nums2,j+1,end2,k-(j-start2+1));

        }
    }
}
