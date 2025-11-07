
import java.util.Locale;
import java.util.Scanner;

class Mahasiswa {

    String nim;
    String nama;
    double ipk;

    public Mahasiswa(String nim, String nama, double ipk) {
        this.nim = nim;
        this.nama = nama;
        this.ipk = ipk;
    }

    public String getNim() {
        return nim;

    }

    public String getNama() {
        return nama;
    }

    public double getIpk() {
        return ipk;
    }

    @Override
    public String toString() {
        return String.format(Locale.US, "%s-%s-%.2f", this.nim, this.nama, this.ipk);
    }
}

class Node<T> {

    T data;
    Node<T> next;

    public Node(T data) {
        this.data = data;
        this.next = null;
    }
}

class SLL {

    Node<Mahasiswa> head, tail;
    int size = 0;

    public boolean isEmpty() {
        return size == 0;
    }

    public void addFirst(Mahasiswa data) {
        Node<Mahasiswa> newNode = new Node<>(data);
        if (isEmpty()) {
            head = tail = newNode;
        } else {
            newNode.next = head;
            head = newNode;
        }
        size++;
    }

    public void addLast(Mahasiswa data) {
        Node<Mahasiswa> newNode = new Node<>(data);
        if (isEmpty()) {
            head = tail = newNode;
        } else {
            tail.next = newNode;
            tail = newNode;

        }
        size++;

    }

    public void removeByNim(String nim) {
        if (isEmpty()) {
            return;
        }
        if (head.data.getNim().equals(nim)) {
            head = head.next;
            if (head == null) {
                tail = null;
            }
            size--;
            return;
        }
        Node<Mahasiswa> current = head;
        while (current.next != null && !current.next.data.getNim().equals(nim)) {
            current = current.next;
        }
        if (current.next != null) {
            if (current.next == tail) {
                tail = current;
            }
            current.next = current.next.next;
            size--;
        }
    }

    public void findByNim(String nim) {
        Node<Mahasiswa> current = head;
        while (current != null) {
            if (current.data.getNim().equals(nim)) {
                System.out.println(current.data.toString());
                return;
            }
            current = current.next;
        }
        System.out.println("Mahasiswa tidak ditemukan");
    }

    public void printAll() {
        if (isEmpty()) {
            System.out.println("List kosong");
            return;
        }
        StringBuilder sb = new StringBuilder();
        Node<Mahasiswa> current = head;
        while (current != null) {
            sb.append("[").append(current.data.toString()).append("]");
            if (current.next != null) {
                sb.append(" -> ");
            }
            current = current.next;
        }
        System.out.println(sb.toString());
    }

    public void findMaxMinIpk() {
        if (isEmpty()) {
            System.out.println("List kosong");
            return;
        }
        Mahasiswa maxMhs = head.data;
        Mahasiswa minMhs = head.data;
        Node<Mahasiswa> current = head.next;
        while (current != null) {
            if (Double.compare(current.data.getIpk(), maxMhs.getIpk()) > 0) {
                maxMhs = current.data;
            }
            if (Double.compare(current.data.getIpk(), minMhs.getIpk()) < 0) {
                minMhs = current.data;
            }
            current = current.next;
        }
        System.out.println("Maksimum: " + maxMhs.toString());
        System.out.println("Minimum: " + minMhs.toString());
    }

    public void printSize() {
        System.out.println(this.size);
    }
}

public class Main {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        SLL list = new SLL();

        if (!scanner.hasNextLine()) {
            scanner.close();
            return;
        }
        String qLine = scanner.nextLine().trim();

        int q;
        try {
            q = Integer.parseInt(qLine);
        } catch (Exception e) {
            scanner.close();
            return;
        }

        for (int i = 0; i < q; i++) {
            if (!scanner.hasNextLine()) {
                break;
            }
            String line = scanner.nextLine().trim();
            if (line.isEmpty()) {
                continue;
            }
            line = line.trim();

            String[] parts = line.split("\\s+");
            String command = parts[0];

            if (command.equals("PRINT") && parts.length >= 2 && parts[1].equalsIgnoreCase("ALL")) {
                command = "PRINT_ALL";
            }

            switch (command) {
                case "ADD_FIRST":
                case "ADD_LAST":
                    if (parts.length < 4) {
                        break;
                    }
                    String nim = parts[1];
                    String nama = parts[2];

                    double ipk;
                    try {
                        ipk = Double.parseDouble(parts[3]);
                    } catch (NumberFormatException ex) {
                        System.out.println("Error: IPK harus antara 0.0 hingga 4.0");
                        break;
                    }
                    if (nim.length() != 10) {
                        System.out.println("Error: NIM harus 1 digit");
                        System.out.println("Error: NIM harus 10 digit");
                        break;
                    }
                    if (ipk < 0.0 || ipk > 4.0) {
                        System.out.println("Error: IPK harus antara 0.0 hingga 4.0");
                        break;
                    }

                    Mahasiswa m = new Mahasiswa(nim, nama, ipk);
                    if (command.equals("ADD_FIRST")) {
                        list.addFirst(m);
                    } else {
                        list.addLast(m);
                    }
                    break;

                case "REMOVE":
                    if (parts.length < 2) {
                        break;
                    }
                    list.removeByNim(parts[1]);
                    break;
                case "FIND":
                    if (parts.length < 2) {
                        break;
                    }
                    list.findByNim(parts[1]);
                    break;
                case "PRINT_ALL":
                    list.printAll();
                    break;
                case "MAX_MIN_IPK":
                    list.findMaxMinIpk();
                    break;
                case "SIZE":
                    list.printSize();
                    break;

                default:
                    break;

            }

        }
        scanner.close();
    }
}
