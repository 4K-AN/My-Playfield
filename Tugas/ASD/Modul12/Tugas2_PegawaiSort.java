import java.util.Scanner;

class Pegawai {
    String nip;
    String nama;

    public Pegawai(String nip, String nama) {
        this.nip = nip;
        this.nama = nama;
    }

    public String toString() {
        return nip + " - " + nama;
    }
}

public class Tugas2_PegawaiSort {
    
    // Insertion Sort untuk Object Pegawai
    public static void sortPegawai(Pegawai[] arr, int byType, int order) {
        // byType: 1=NIP, 2=Nama
        // order: 1=Asc, 2=Desc
        
        int n = arr.length;
        for (int i = 1; i < n; i++) {
            Pegawai key = arr[i];
            int j = i - 1;
            boolean swapCondition = false;

            while (j >= 0) {
                // Tentukan logika perbandingan
                int compareResult;
                if (byType == 1) { // By NIP
                    compareResult = arr[j].nip.compareTo(key.nip);
                } else { // By Nama
                    compareResult = arr[j].nama.compareTo(key.nama);
                }

                // Tentukan Ascending atau Descending
                if (order == 1) { // Ascending (Normal)
                    swapCondition = (compareResult > 0);
                } else { // Descending (Terbalik)
                    swapCondition = (compareResult < 0);
                }

                if (swapCondition) {
                    arr[j + 1] = arr[j];
                    j--;
                } else {
                    break;
                }
            }
            arr[j + 1] = key;
        }
    }

    public static void tampil(Pegawai[] arr) {
        for (Pegawai p : arr) {
            System.out.println(p);
        }
    }

    public static void main(String[] args) {
        Pegawai[] data = {
            new Pegawai("105", "Budi"),
            new Pegawai("102", "Andi"),
            new Pegawai("108", "Citra"),
            new Pegawai("101", "Doni"),
            new Pegawai("103", "Eka")
        };

        Scanner sc = new Scanner(System.in);
        System.out.println("Data Awal:");
        tampil(data);

        System.out.println("\n--- Menu Sorting Pegawai ---");
        System.out.println("1. Berdasarkan NIP");
        System.out.println("2. Berdasarkan Nama");
        System.out.print("Pilih: ");
        int type = sc.nextInt();

        System.out.println("1. Ascending (Naik)");
        System.out.println("2. Descending (Turun)");
        System.out.print("Pilih: ");
        int order = sc.nextInt();

        sortPegawai(data, type, order);

        System.out.println("\nData Terurut:");
        tampil(data);
    }
}