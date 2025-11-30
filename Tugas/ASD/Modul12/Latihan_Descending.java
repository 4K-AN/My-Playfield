public class Latihan_Descending {
    
    // --- BUBBLE SORT DESCENDING ---
    public static void bubbleSortDesc(int[] arr) {
        System.out.println("\n=== Bubble Sort Descending ===");
        int n = arr.length;
        int compareCount = 0;
        int swapCount = 0;

        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                compareCount++;
                // Ubah > menjadi < untuk Descending
                if (arr[j] < arr[j + 1]) { 
                    // Swap
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                    swapCount++;
                }
            }
            // Tampilkan Iterasi
            System.out.print("Iterasi " + (i+1) + ": ");
            printArray(arr);
        }
        System.out.println("Total Perbandingan: " + compareCount);
        System.out.println("Total Pergeseran (Swap): " + swapCount);
    }

    // --- SELECTION SORT DESCENDING ---
    public static void selectionSortDesc(int[] arr) {
        System.out.println("\n=== Selection Sort Descending ===");
        int n = arr.length;
        int compareCount = 0;
        int swapCount = 0;

        for (int i = 0; i < n - 1; i++) {
            int maxIdx = i; // Kita cari elemen TERBESAR untuk ditaruh di depan
            for (int j = i + 1; j < n; j++) {
                compareCount++;
                if (arr[j] > arr[maxIdx]) { // Cari max
                    maxIdx = j;
                }
            }

            // Swap jika ditemukan yang lebih besar
            if (maxIdx != i) {
                int temp = arr[maxIdx];
                arr[maxIdx] = arr[i];
                arr[i] = temp;
                swapCount++;
            }
            
            // Tampilkan Iterasi
            System.out.print("Iterasi " + (i+1) + ": ");
            printArray(arr);
        }
        System.out.println("Total Perbandingan: " + compareCount);
        System.out.println("Total Pergeseran (Swap): " + swapCount);
    }

    public static void printArray(int[] arr) {
        for (int val : arr) System.out.print(val + " ");
        System.out.println();
    }

    public static void main(String[] args) {
        int[] data1 = {10, 5, 20, 8, 2, 15};
        int[] data2 = {10, 5, 20, 8, 2, 15};

        bubbleSortDesc(data1);
        selectionSortDesc(data2);
    }
}