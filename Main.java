import java.util.Scanner;

public class Main {

    // Konversi nilai huruf ke angka IPK (double)
    // Jika nilai tidak valid -> tampilkan pesan dan kembalikan 0.00
    public static double konversiNilai(String hurufRaw) {
        if (hurufRaw == null) return 0.0;
        String original = hurufRaw;
        String huruf = hurufRaw.trim().replaceAll("\\s+", "").toUpperCase();

        switch (huruf) {
            case "A":
                return 4.00;
            case "B+":
                return 3.50;
            case "B":
                return 3.00;
            case "C+":
                return 2.50;
            case "C":
                return 2.00;
            case "D+":
                return 1.50;
            case "D":
                return 1.00;
            case "E":
            case "K":
                return 0.00;
            default:
                System.out.println("Nilai tidak valid: " + original + " (dianggap 0.00)");
                return 0.00;
        }
    }

    // Hitung rata dari lima nilai double
    public static double hitungRata(double a, double b, double c, double d, double e) {
        return (a + b + c + d + e) / 5.0;
    }

    // Cari nilai minimum dari lima nilai double
    public static double cariMin(double a, double b, double c, double d, double e) {
        double min = Math.min(a, b);
        min = Math.min(min, c);
        min = Math.min(min, d);
        min = Math.min(min, e);
        return min;
    }

    // Cari nilai maksimum dari lima nilai double
    public static double cariMax(double a, double b, double c, double d, double e) {
        double max = Math.max(a, b);
        max = Math.max(max, c);
        max = Math.max(max, d);
        max = Math.max(max, e);
        return max;
    }

    // Tampilkan hasil sesuai format (perhatikan jumlah karakter sampai ':')
    public static void tampilkanHasil(double rata, double min, double max) {
        System.out.println("=== HASIL PERHITUNGAN IPK ===");
        // Label dibuat supaya dari huruf pertama sampai ':' berjumlah 16 karakter
        System.out.println("IPK Rata-rata  : " + String.format("%.2f", rata));
        System.out.println("IPK Tertinggi  : " + max);
        System.out.println("IPK Terendah   : " + min);

        String status = (rata >= 2.75) ? "LULUS" : "TIDAK LULUS";
        System.out.println("Status: " + status);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        // Baca 5 nilai (token dipisah spasi atau newline)
        String n1 = sc.hasNext() ? sc.next() : "";
        String n2 = sc.hasNext() ? sc.next() : "";
        String n3 = sc.hasNext() ? sc.next() : "";
        String n4 = sc.hasNext() ? sc.next() : "";
        String n5 = sc.hasNext() ? sc.next() : "";

        double v1 = konversiNilai(n1);
        double v2 = konversiNilai(n2);
        double v3 = konversiNilai(n3);
        double v4 = konversiNilai(n4);
        double v5 = konversiNilai(n5);

        double rata = hitungRata(v1, v2, v3, v4, v5);
        double min = cariMin(v1, v2, v3, v4, v5);
        double max = cariMax(v1, v2, v3, v4, v5);

        tampilkanHasil(rata, min, max);

        sc.close();
    }
}

