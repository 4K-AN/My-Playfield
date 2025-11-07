
public class Main {

    public static void main(String[] args) {

        SLL<Mahasiswa> daftarMahasiswa = new SLL<>();

        Mahasiswa mhs1 = new Mahasiswa("23001", "Budi", 3.5);
        Mahasiswa mhs2 = new Mahasiswa("23002", "Siti", 3.9);
        Mahasiswa mhs3 = new Mahasiswa("23003", "Joko", 3.2);
        Mahasiswa mhs4 = new Mahasiswa("23004", "Ani", 4.0);
        Mahasiswa mhs5 = new Mahasiswa("23005", "Rina", 3.7);

        System.out.println("Menambahkan data Mahasiswa secara terurut berdasarkan IPK...");
        daftarMahasiswa.addSortedByIpk(mhs1);
        daftarMahasiswa.addSortedByIpk(mhs2);
        daftarMahasiswa.addSortedByIpk(mhs3);
        daftarMahasiswa.addSortedByIpk(mhs4);
        daftarMahasiswa.addSortedByIpk(mhs5);

        System.out.println("\n============================================================");
        System.out.println("Daftar Akhir Mahasiswa (Urut Berdasarkan IPK Tertinggi):");
        System.out.println("============================================================");
        daftarMahasiswa.print();
    }
}
