
public class SLL<T> {

    Node<T> head, tail;
    int size = 0;

    public SLL() {
        head = null;
        tail = null;
        size = 0;
    }

    public boolean isEmpty() {
        return size == 0;
    }

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

    public void print() {
        if (isEmpty()) {
            System.out.println("List Mahasiswa Kosong");
        } else {
            Node<T> temp = head;
            int i = 1;
            while (temp != null) {
                System.out.println((i++) + ". " + temp.data.toString());
                temp = temp.next;
            }
        }
    }

    public void addSortedByIpk(Mahasiswa data) {
        Node<Mahasiswa> newNode = new Node<>(data);
        if (isEmpty() || newNode.data.getIpk() >= ((Mahasiswa) head.data).getIpk()) {
            addFirst((T) data);
            return;
        }
        Node<Mahasiswa> current = (Node<Mahasiswa>) head;
        while (current.next != null && current.next.data.getIpk() > newNode.data.getIpk()) {
            current = current.next;
        }
        newNode.next = current.next;
        current.next = newNode;
        if (newNode.next == null) {
            tail = (Node<T>) newNode;
        }
        size++;
    }
}
