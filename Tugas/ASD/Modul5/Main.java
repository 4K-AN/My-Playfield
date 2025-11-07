
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

    // 1. Inisialisasi 
    public boolean isEmpty() {
        return size == 0;
    }

    // 3. size
    public int size() {
        return size;
    }

    // 4. Penambahan
    public void addFirst(T data) {
        Node<T> newNode = new Node<>(data);
        if (isEmpty()) {
            head = tail = newNode;
        } else {
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
        } else {
            tail.next = newNode;
            newNode.prev = tail;
            tail = newNode;
        }
        size++;
    }

    // 5. Penghapusan
    public void removeFirst() {
        if (isEmpty()) {
            return;
        }
        if (size == 1) {
            head = tail = null;
        } else {
            head = head.next;
            head.prev = null;
        }
        size--;
    }

    public void removeLast() {
        if (isEmpty()) {
            return;
        }
        if (size == 1) {
            head = tail = null;
        } else {
            tail = tail.prev;
            tail.next = null;
        }
        size--;
    }

    // 6. Penyisipan
    public void insertAt(int index, T data) {
        if (index < 0 || index > size) {
            return;
        }
        if (index == 0) {
            addFirst(data);
            return;
        }
        if (index == size) {
            addLast(data);
            return;
        }
        Node<T> current = head;
        for (int i = 0; i < index; i++) {
            current = current.next;
        }
        Node<T> newNode = new Node<>(data);
        newNode.prev = current.prev;
        newNode.next = current;
        current.prev.next = newNode;
        current.prev = newNode;
        size++;
    }

    // 7. Pencarian
    public int indexOf(T data) {
        Node<T> current = head;
        int index = 0;
        while (current != null) {
            if (current.data.equals(data)) {
                return index;
            }
            current = current.next;
            index++;
        }
        return -1;
    }

    // 8. Pengaksesan
    public T get(int index) {
        if (index < 0 || index >= size) {
            return null;
        }
        Node<T> current = head;
        for (int i = 0; i < index; i++) {
            current = current.next;
        }
        return current.data;
    }

    public void printForward() {
        if (isEmpty()) {
            System.out.println("List Kosong");
            return;
        }
        Node<T> current = head;
        while (current != null) {
            System.out.print(current.data + " -> ");
            current = current.next;
        }
        System.out.println("null");
    }
}

public class Main {

    public static void main(String[] args) {
        System.out.println("--- DEMO LATIHAN 1: DLL LENGKAP ---");
        DLL<String> listNama = new DLL<>();
        listNama.addLast("Budi");
        listNama.addLast("Siti");
        listNama.addFirst("Ani");
        System.out.print("Isi awal: ");
        listNama.printForward();

        System.out.println("\nMenyisipkan 'Rina' di indeks ke-2 (Penyisipan):");
        listNama.insertAt(2, "Rina");
        listNama.printForward();

        System.out.println("\nMengambil data di indeks ke-1 (Pengaksesan): " + listNama.get(1)); // Budi
        System.out.println("Mencari data 'Siti' (Pencarian): Berada di indeks " + listNama.indexOf("Siti")); // 3

        System.out.println("\nMenghapus data terakhir (Penghapusan):");
        listNama.removeLast();
        listNama.printForward();
    }
}
