
import java.util.Objects;

public class SkibidiLinkedList<T> {

    private static class Node<T> {

        Node<T> prev;
        Node<T> next;
        T data;

        Node(T data) {
            this.data = data;
        }
    }

    private Node<T> head;
    private Node<T> tail;
    private int size = 0;

    public void tambahAkhir(T data) {
        Node<T> n = new Node<>(data);
        if (head == null) {
            head = tail = n;
        } else {
            tail.next = n;
            n.prev = tail;
            tail = n;
        }
        size++;
    }

    public void tambahDepan(T data) {
        Node<T> n = new Node<>(data);
        if (head == null) {
            head = tail = n;
        } else {
            n.next = head;
            head.prev = n;
            head = n;
        }
        size++;
    }

    public void tambahTengah(T data, int posisi) {
        if (posisi < 0 || posisi > size) {
            System.out.println("Posisi tidak valid: " + posisi);
            return;
        }

        if (posisi == 0) {
            tambahDepan(data);
            return;
        }

        if (posisi == size) {
            tambahAkhir(data);
            return;
        }

        Node<T> cursor = head;
        for (int i = 0; i < posisi - 1; i++) {
            cursor = cursor.next;
        }

        Node<T> n = new Node<>(data);
        n.next = cursor.next;
        n.prev = cursor;
        cursor.next.prev = n;
        cursor.next = n;
        size++;
    }

    public void hapusNodeX(T data) {
        if (head == null) {
            System.out.println("Linked list kosong");
            return;
        }

        Node<T> cursor = head;
        while (cursor != null && !Objects.equals(cursor.data, data)) {
            cursor = cursor.next;
        }

        if (cursor == null) {
            System.out.println("Data tidak ditemukan: " + data);
            return;
        }

        if (cursor == head) {
            head = head.next;
            if (head != null) {
                head.prev = null;
            } else {

                tail = null;
            }
            size--;
            return;
        }

        if (cursor == tail) {
            tail = tail.prev;
            if (tail != null) {
                tail.next = null;
            } else {
                head = null;
            }
            size--;
            return;
        }

        cursor.prev.next = cursor.next;
        cursor.next.prev = cursor.prev;
        size--;
    }

    public void cetak() {
        if (head == null) {
            System.out.println("Linked list kosong");
            return;
        }

        Node<T> cursor = head;
        while (cursor != null) {
            System.out.println(cursor.data);
            cursor = cursor.next;
        }
    }

    public void cetakMundur() {
        if (tail == null) {
            System.out.println("Linked list kosong");
            return;
        }

        Node<T> cursor = tail;
        while (cursor != null) {
            System.out.println(cursor.data);
            cursor = cursor.prev;
        }
    }

    public int size() {
        return size;
    }

    public static void main(String[] args) {
        SkibidiLinkedList<String> list = new SkibidiLinkedList<>();
        list.tambahAkhir("A");
        list.tambahAkhir("B");
        list.tambahAkhir("C");
        list.tambahDepan("Z");
        list.tambahTengah("X", 2);

        System.out.println("Cetak maju:");
        list.cetak();

        System.out.println("\nCetak mundur:");
        list.cetakMundur();

        System.out.println("\nHapus X:");
        list.hapusNodeX("X");
        list.cetak();

        System.out.println("\nUkuran sekarang: " + list.size());
    }
}
