
public class SLL<T> {

    Node<T> head, tail;
    int size = 0;

    // 1. Inisialisasi (dilakukan oleh constructor)
    public SLL() {
        head = null;
        tail = null;
        size = 0;
    }

    // 2. isEmpty
    public boolean isEmpty() {
        return size == 0;
    }

    // 3. size
    public int size() {
        return size;
    }

    // 4. Penambahan (Add)
    public void addFirst(T data) {
        Node<T> newNode = new Node<>(data);
        if (isEmpty()) {
            head = tail = newNode;
        } else {
            newNode.next = head;
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
            tail = newNode;
        }
        size++;
    }

    // 5. Penyisipan (Insert)
    public void insertAt(int index, T data) {
        if (index < 0 || index > size) {
            System.out.println("Error: Index di luar batas");
            return;
        }
        if (index == 0) {
            addFirst(data);
        } else if (index == size) {
            addLast(data);
        } else {
            Node<T> newNode = new Node<>(data);
            Node<T> temp = head;
            for (int i = 0; i < index - 1; i++) {
                temp = temp.next;
            }
            newNode.next = temp.next;
            temp.next = newNode;
            size++;
        }
    }

    // 6. Penghapusan (Remove)
    public void removeFirst() {
        if (!isEmpty()) {
            head = head.next;
            if (head == null) {
                tail = null;
            }
            size--;
        } else {
            System.out.println("List Kosong, tidak ada yang bisa dihapus");
        }
    }

    public void removeLast() {
        if (isEmpty()) {
            System.out.println("List Kosong, tidak ada yang bisa dihapus");
        } else if (size == 1) {
            head = tail = null;
            size--;
        } else {
            Node<T> temp = head;
            while (temp.next != tail) {
                temp = temp.next;
            }
            temp.next = null;
            tail = temp;
            size--;
        }
    }

    public void removeAt(int index) {
        if (index < 0 || index >= size) {
            System.out.println("Error: Index di luar batas");
            return;
        }
        if (index == 0) {
            removeFirst();
        } else {
            Node<T> temp = head;
            for (int i = 0; i < index - 1; i++) {
                temp = temp.next;
            }
            temp.next = temp.next.next;
            if (temp.next == null) {
                tail = temp;
            }
            size--;
        }
    }

    // 7. Pencarian (Search)
    public int indexOf(T data) {
        Node<T> temp = head;
        int index = 0;
        while (temp != null) {
            if (temp.data.equals(data)) {
                return index;
            }
            temp = temp.next;
            index++;
        }
        return -1; // Data tidak ditemukan
    }

    // 8. Pengaksesan (Access)
    public T get(int index) {
        if (index < 0 || index >= size) {
            System.out.println("Error: Index di luar batas");
            return null;
        }
        Node<T> temp = head;
        for (int i = 0; i < index; i++) {
            temp = temp.next;
        }
        return temp.data;
    }

    // Method bantuan untuk mencetak list
    public void print() {
        if (isEmpty()) {
            System.out.println("List Kosong");
        } else {
            Node<T> temp = head;
            while (temp != null) {
                System.out.print(temp.data + " -> ");
                temp = temp.next;
            }
            System.out.println("null");
        }
    }
}
