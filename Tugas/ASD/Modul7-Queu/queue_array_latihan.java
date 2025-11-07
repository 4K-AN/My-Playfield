import java.util.Scanner;

public class queue_array_latihan {
    Scanner masuk = new Scanner(System.in);
    int choice, i;
    char item;

    static final int MAX_SIZE = 10;
    char arr_queue[] = new char[MAX_SIZE];
    int keluar = 0;
    
    int rear = 0; 

    
    public void enqueue(char item) {
        if (rear == MAX_SIZE) {
            System.out.print("\n# Queue Penuh");
        } else {
            
            for (i = rear; i > 0; i--) {
                arr_queue[i] = arr_queue[i - 1];
            }

            
            arr_queue[0] = item;
            
            
            rear++;
            
            System.out.print("\n# Enqueue item: " + item + " ke index 0");
        }
    }

    
    public void dequeue() {
        if (rear == 0) {
            System.out.print("\n## Queue kosong");
        } else {
            
            char dequeuedItem = arr_queue[rear - 1];
            
            
            rear--;
            
            System.out.print("\n## Dequeue Value:" + dequeuedItem + " (dari index " + rear + ")");
        }
    }

    
    public void printAll() {
        System.out.print("\n## Queue Size: " + rear);
        System.out.print("\n## Isi Queue (Index 0 (Front) -> Index " + (rear-1) + " (Rear)): ");
        
        if (rear == 0) {
            System.out.print("[]");
            return;
        }

        System.out.print("[");
        for (i = 0; i < rear; i++) {
            System.out.print(arr_queue[i]);
            if (i < rear - 1) {
                System.out.print(", ");
            }
        }
        System.out.print("]");
    }

    
    public void menu() {
        System.out.print("\n\n(Latihan 8.9) Operasi (1:enqueue, 2:dequeue, 3:print, 4:exit): ");
        choice = masuk.nextInt();
        switch (choice) {
            case 1:
                System.out.print("\nMasukkan huruf yang akan di-enqueue: ");
                item = masuk.next().charAt(0);
                enqueue(item);
                break;
            case 2:
                dequeue();
                break;
            case 3:
                printAll();
                break;
            default:
                System.out.print("\nKeluar program.\n");
                keluar = 1;
                break;
        }
    }

    
    public static void main(String[] args) {
        queue_array_latihan queue = new queue_array_latihan();
        do {
            queue.menu();
        } while (queue.keluar == 0);
    }
}