import java.util.Random;

public class ShellSorter {
    
    // Menggunakan Integer agar kompatibel dengan compareTo
    public static void shellSort(Integer[] arr) {
        int i, jarak;
        boolean did_swap = true;
        Integer temp;
        jarak = arr.length;
        
        while (jarak > 1) {
            jarak = jarak / 2;
            did_swap = true;
            while (did_swap) {
                did_swap = false;
                i = 0;
                while (i < (arr.length - jarak)) {
                    if (arr[i].compareTo(arr[i + jarak]) > 0) {
                        temp = arr[i];
                        arr[i] = arr[i + jarak];
                        arr[i + jarak] = temp;
                        did_swap = true;
                    }
                    i++;
                }
            }
        }
    }

    public static void tampil(Integer[] data) {
        for (Integer objek : data) {
            System.out.print(objek + " ");
        }
        System.out.println("");
    }

    public static void main(String[] args) {
        Integer data[] = new Integer[10];
        Random rand = new Random();
        
        // Mengisi data acak
        for (int a = 0; a < data.length; a++) {
            data[a] = rand.nextInt(100); // Angka acak 0-99
        }

        System.out.println("Sebelum diurutkan:");
        tampil(data);
        
        shellSort(data);
        
        System.out.println("Sesudah diurutkan:");
        tampil(data);
    }
}