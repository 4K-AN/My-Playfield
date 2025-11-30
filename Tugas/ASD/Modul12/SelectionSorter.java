public class SelectionSorter {
    int[] L = {25, 27, 10, 8, 76, 21};

    void selectionSort() {
        int j, k, i, temp;
        int jmax, u = 5; // u adalah batas indeks terakhir (length - 1)
        
        for (j = 0; j < 6; j++) {
            jmax = 0;
            System.out.println("Langkah " + (j + 1) + ":");
            
            // Mencari nilai maksimum
            for (k = 1; k <= u; k++) {
                if (L[k] > L[jmax]) {
                    jmax = k;
                }
            }
            
            // Menukar nilai maksimum ke posisi belakang (posisi u)
            temp = L[u];
            L[u] = L[jmax];
            L[jmax] = temp;
            
            u--; // Kurangi batas array yang belum terurut
            
            // Melihat hasil tiap langkah
            for (i = 0; i <= 5; i++) {
                System.out.print(L[i] + " ");
            }
            System.out.println();
        }
        
        System.out.println("Hasil akhir:");
        for (i = 0; i <= 5; i++) {
            System.out.println(L[i] + " index:" + (i + 1));
        }
    }

    public static void main(String[] args) {
        SelectionSorter sorter = new SelectionSorter();
        sorter.selectionSort();
    }
}