import java.util.Arrays;

public class PengolahanArray {

    public static boolean adalahPrima(int angka) {

        if (angka <= 1) {
            return false;
        }

        for (int i = 2; i < angka; i++) {
            if (angka % i == 0) {
                return false;
            }
        }
        return true;
    }

    public static void main(String[] args) {

        int[] data = {30, 87, 90, 3, 1, 50, 23, 4, 25, 23, 40, 35, 47, 2, 33};

        System.out.println("--- LATIHAN: DATA AWAL ---");

        System.out.println(Arrays.toString(data));
        System.out.println("========================================\n");

        System.out.println("--- TUGAS: HASIL PENGOLAHAN ---");

        Arrays.sort(data);
        System.out.println("1. Data setelah diurutkan:");
        System.out.println(Arrays.toString(data) + "\n");

        double total = 0;
        for (int i = 0; i < data.length; i++) {
            total = total + data[i];
        }
        double rataRata = total / data.length;
        System.out.println("2. Nilai rata-rata data:");
        System.out.println("   Total: " + total);
        System.out.println("   Jumlah Data: " + data.length);
        System.out.println("   Rata-rata: " + rataRata + "\n");

        int nilaiMin = data[0];
        int nilaiMax = data[data.length - 1];
        System.out.println("3. Nilai minimal dan maksimal:");
        System.out.println("   Nilai Minimal: " + nilaiMin);
        System.out.println("   Nilai Maksimal: " + nilaiMax + "\n");

        System.out.println("4. Data bilangan ganjil dan prima:");
        System.out.print("   Bilangan Ganjil: ");
        for (int i = 0; i < data.length; i++) {
            if (data[i] % 2 != 0) {
                System.out.print(data[i] + " ");
            }
        }
        System.out.println();

        System.out.print("   Bilangan Prima: ");
        for (int i = 0; i < data.length; i++) {
            if (adalahPrima(data[i])) {
                System.out.print(data[i] + " ");
            }
        }
        System.out.println("\n");

        int baris = 3;
        int kolom = 5;
        int[][] data2D = new int[baris][kolom];
        int indexDataSatuDimensi = 0;

        for (int i = 0; i < baris; i++) {
            for (int j = 0; j < kolom; j++) {
                data2D[i][j] = data[indexDataSatuDimensi];
                indexDataSatuDimensi++;
            }
        }

        System.out.println("5. Array 2 Dimensi (format 3 baris dan 5 kolom):");
        for (int i = 0; i < baris; i++) {
            System.out.println("   " + Arrays.toString(data2D[i]));
        }
    }
}
