import java.util.Random;

public class Tugas1_RandomAnalysis {
    
    public static void insertionSortCount(int[] arr) {
        int iterations = 0;
        int n = arr.length;
        for (int i = 1; i < n; i++) {
            int key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
                iterations++; // Hitung pergeseran dalam while
            }
            arr[j + 1] = key;
        }
        System.out.println("Insertion Sort - Jumlah Iterasi (Pergeseran): " + iterations);
    }

    public static void bubbleSortCount(int[] arr) {
        int iterations = 0;
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                iterations++; // Hitung setiap perbandingan
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
        System.out.println("Bubble Sort - Jumlah Iterasi (Perbandingan): " + iterations);
    }

    public static void main(String[] args) {
        int[] data1 = new int[50];
        int[] data2 = new int[50];
        Random rand = new Random();

        // Isi 50 data acak
        for(int i=0; i<50; i++) {
            int num = rand.nextInt(100);
            data1[i] = num;
            data2[i] = num; // Copy data agar adil
        }

        System.out.println("Melakukan Sorting pada 50 Data Acak...");
        insertionSortCount(data1);
        bubbleSortCount(data2);
    }
}