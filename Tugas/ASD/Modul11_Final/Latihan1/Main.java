
import java.util.*;

public class Main {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        int n = sc.nextInt();
        sc.nextLine();
        
        Queue<String> antrian = new LinkedList<>();
        
        for (int i = 0; i < n; i++) {
            String line = sc.nextLine().trim();
            String[] parts = line.split("\\s+");
            String cmd = parts[0].toUpperCase();
            
            if (cmd.equals("DAFTAR")) {
                if (parts.length >= 2) {
                    String nama = parts[1];
                    antrian.offer(nama);
                }
            } else if (cmd.equals("PROSES")) {
                if (antrian.isEmpty()) {
                    System.out.println("Antrian kosong");
                } else {
                    String nama = antrian.poll();
                    System.out.println(nama + " mendapatkan unit");
                }
            } else if (cmd.equals("LIHAT")) {
                if (antrian.isEmpty()) {
                    System.out.println("Antrian kosong");
                } else {
                    String nama = antrian.peek();
                    System.out.println("Pelanggan terdepan: " + nama);
                }
            }
        }
        
        sc.close();
    }
}