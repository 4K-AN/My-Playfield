
import java.util.Locale;

class Mahasiswa {

    String nim;
    String nama;
    double ipk;

    public Mahasiswa(String nim, String nama, double ipk) {
        this.nim = nim;
        this.nama = nama;
        this.ipk = ipk;
    }

    public double getIpk() {
        return ipk;
    }

    @Override
    public String toString() {
        return String.format(Locale.US, "Mahasiswa{nim='%s', nama='%s', ipk=%.2f}", nim, nama, ipk);
    }
}
// Class Node dan DLL dari Latihan 1 disalin kembali di sini untuk kemudahan.
// Class Node dan DLL dari Latihan 1 disalin kembali di sini untuk kemudahan.
class Node<T> {

    T data;
    Node<T> next;
    Node<T> prev;

    public Node(T data) {
        this.data = data;
        this.next = null;
        this.prev = null;
    }
}

class DLL<T> {

    Node<T> head, tail;
    int size = 0;

    public boolean isEmpty() {
        return size == 0;
    }

    public void addFirst(T data) {
        Node<T> newNode = new Node<>(data);
        if (isEmpty()) {
            head = tail = newNode; 
        }else {
            newNode.next = head;
            head.prev = newNode;
            head = newNode;
        }
        size++;
    }

    public void addLast(T data) {
        Node<T> newNode = new Node<>(data);
        if (isEmpty()) {
            head = tail = newNode; 
        }else {
            tail.next = newNode;
            newNode.prev = tail;
            tail = newNode;
        }
        size++;
    }

    public void printForward() {
        if (isEmpty()) {
            System.out.println("List Kosong");
            return;
        }
        Node<T> current = head;
        int i = 1;
        while (current != null) {
            System.out.println((i++) + ". " + current.data.toString());
            current = current.next;
        }
    }

    public void printBackward() {
        if (isEmpty()) {
            System.out.println("List Kosong");
            return;
        }
        Node<T> current = tail;
        int i = size;
        while (current != null) {
            System.out.println((i--) + ". " + current.data.toString());
            current = current.prev;
        }
    }

    @SuppressWarnings("unchecked")
    public void addSortedByIpk(Mahasiswa data) {
        Node<Mahasiswa> newNode = new Node<>(data);
        if (isEmpty() || data.getIpk() >= ((Mahasiswa) head.data).getIpk()) {
            addFirst((T) data);
            return;
        }
        if (data.getIpk() <= ((Mahasiswa) tail.data).getIpk()) {
            addLast((T) data);
            return;
        }
        Node<Mahasiswa> current = (Node<Mahasiswa>) head;
        while (data.getIpk() < current.data.getIpk()) {
            current = current.next;
        }
        newNode.next = current;
        newNode.prev = current.prev;
        current.prev.next = newNode;
        current.prev = newNode;
        size++;
    }

    public void printDescendingByIpk() {
        System.out.println("Data Mahasiswa (Urut IPK Descending):");
        printForward();
    }

    public void printAscendingByIpk() {
        System.out.println("Data Mahasiswa (Urut IPK Ascending):");
        printBackward();
    }
}

public class Main {

    public static void main(String[] args) {
        DLL<Mahasiswa> daftarMahasiswa = new DLL<>();

        Mahasiswa mhs1 = new Mahasiswa("23001", "Budi", 3.5);
        Mahasiswa mhs2 = new Mahasiswa("23002", "Siti", 3.9);
        Mahasiswa mhs3 = new Mahasiswa("23003", "Joko", 3.2);
        Mahasiswa mhs4 = new Mahasiswa("23004", "Ani", 4.0);
        Mahasiswa mhs5 = new Mahasiswa("23005", "Rina", 3.7);

        System.out.println("Menambahkan data Mahasiswa...");
        daftarMahasiswa.addSortedByIpk(mhs1);
        daftarMahasiswa.addSortedByIpk(mhs2);
        daftarMahasiswa.addSortedByIpk(mhs3);
        daftarMahasiswa.addSortedByIpk(mhs4);
        daftarMahasiswa.addSortedByIpk(mhs5);

        System.out.println("\n-------------------------------------------------------");
        daftarMahasiswa.printDescendingByIpk();

        System.out.println("\n-------------------------------------------------------");
        daftarMahasiswa.printAscendingByIpk();
    }
}
