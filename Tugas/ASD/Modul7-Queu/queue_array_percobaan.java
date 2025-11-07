import java.util.Scanner;
public class queue_array_percobaan {
    Scanner masuk = new Scanner(System.in);
    int choice, i;
    char item;

    static final int MAX_SIZE = 10;
    char arr_stack[] = new char[MAX_SIZE];
    int keluar = 0;
    int rear = 0; 
    public void enqueue(char item) {
        if (rear == MAX_SIZE) {
            System.out.print("\n# Queue Penuh");
        } else {
            System.out.print("\n# Queue No urut/index: " + rear + ", Queue:" + item);
            arr_stack[rear++] = item;
            
        }
    }

    
    public void dequeue() {
        if (rear == 0) {
            System.out.print("\n## Queue kosong");
        } else {
            
            
            System.out.print("\n##Dequeue Value:" + arr_stack[0]);

            
            for (i = 1; i < rear; i++) {
                char temp = arr_stack[i];
                arr_stack[i - 1] = temp;
            }
            
            
            rear--;

        }
    }

    
    public void printAll() {
        System.out.print("\n## Queue Size: " + rear);
        for (i = 0; i < rear; i++) {
            System.out.print("\n## No Urut/index: " + i + ", Value:" + arr_stack[i]);
        }
    }

    
    public void menu() {
        System.out.print("\n\nMasukkan operasi yang akan dilakukan (1:enqueue, 2:dequeue, 3:print): ");
        choice = masuk.nextInt();
        switch (choice) {
            case 1: {
                System.out.print("\nMasukkan huruf yang akan di-enqueue: ");
                
            
                item = masuk.next().charAt(0);
                
                enqueue(item);
                break;
            }
            case 2:
                dequeue();
                break;
            case 3:
                printAll();
                break;
            default:
                System.out.print("\n1:enqueue, 2:dequeue, 3:print\n");
                keluar = 1; 
                break;
        }
    }

    
    public static void main(String[] args) {
        
        queue_array_percobaan queue = new queue_array_percobaan(); 
        
        do {
            queue.menu();
        } while (queue.keluar == 0); 
    }
}