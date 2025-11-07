import java.util.Scanner;


public class stack_array {
    
   
    static final int MAX_SIZE = 10;
    
    Scanner masuk = new Scanner(System.in);
    int choice, i;
    char arr_stack[] = new char[MAX_SIZE]; 
    int count = 0;
    int keluar = 0;

    public void push(char item) {
        if (count == MAX_SIZE) {
            System.out.print("\n# Stack Penuh");
        } else {
     
            arr_stack[count] = item;
            System.out.print("\n# PUSH No urut/index : " + count + ", Push :" + item);
            count++;
           
        }
    }

    public void pop() {
        if (count == 0)
            System.out.print("\n## Stack kosong");
        else {
            // --- Kode tambahan dari 7.5 (3) ---
            --count;
            System.out.print("\n##POP No urut/index : " + count + ", Value :" + arr_stack[count]);
            // ---------------------------------
        }
    }

    public void printAll() {
        System.out.print("\n## Stack Size : " + count);
        for (i = (count - 1); i >= 0; i--)
            System.out.print("\n## No Urut/index : " + i + ", Value :" + arr_stack[i]);
    }

    public void menu() {
        System.out.print("\nMasukkan operasi yang akan dilakukan (1:push, 2:pop, 3:print) : ");
        choice = masuk.nextInt();
        switch (choice) {
            case 1: {
                System.out.print("\nMasukkan huruf yang akan dipush : ");
                // Mengambil input char dengan benar
                char item = masuk.next().charAt(0);
                push(item);
            }
                break;
            case 2:
                pop();
                break;
            case 3:
                printAll();
                break;
            default:
                System.out.print("\n1:push, 2:pop, 3:print\n");
                keluar = 1;
                break;
        }
    }

    // Main method yang benar
    public static void main(String[] args) {
        // Membuat instance dari class untuk memanggil method non-static
        stack_array s = new stack_array();
        do {
            s.menu();
        } while (s.keluar == 0);
    }
}