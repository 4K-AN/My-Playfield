public class InsertionSorter {
    // Size 7 agar index 1 sampai 6 bisa dipakai (index 0 diabaikan sesuai modul)
    int[] L = new int[7]; 

    void insertionSort() {
        int k, temp, j;
        L[1] = 29; L[2] = 27; L[3] = 10;
        L[4] = 8;  L[5] = 76; L[6] = 21;

        for (k = 2; k <= 6; k++) {
            temp = L[k];
            j = k - 1;
            
            System.out.println("Langkah " + (k - 1));
            
            // Geser elemen yang lebih besar dari temp ke kanan
            while (j >= 1 && temp < L[j]) { // Modifikasi kondisi agar tidak out of bound
                L[j + 1] = L[j];
                j--;
            }
            
            L[j + 1] = temp;

            // Cetak per langkah
            for (int i = 1; i <= 6; i++) {
                System.out.println(L[i] + " index:" + i);
            }
        }
    }

    public static void main(String[] args) {
        InsertionSorter sorter = new InsertionSorter();
        sorter.insertionSort();
    }
}