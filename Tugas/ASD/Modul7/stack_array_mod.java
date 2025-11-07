
import java.util.Scanner;

public class stack_array_mod {

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

            for (i = count; i > 0; i--) {
                arr_stack[i] = arr_stack[i - 1];
            }

            arr_stack[0] = item;
            System.out.print("\n# PUSH No urut/index : 0, Push :" + item);
            count++;
        }
    }

    public void pop() {
        if (count == 0) {
            System.out.print("\n## Stack kosong");
        } else {

            char itemPopped = arr_stack[0];
            System.out.print("\n##POP No urut/index : 0, Value :" + itemPopped);

            for (i = 0; i < count - 1; i++) {
                arr_stack[i] = arr_stack[i + 1];
            }

            count--;
        }
    }

    public void printAll() {
        System.out.print("\n## Stack Size : " + count);
        for (i = 0; i < count; i++) {
            System.out.print("\n## No Urut/index : " + i + ", Value :" + arr_stack[i]);
        }
    }

    public void menu() {
        System.out.print("\nMasukkan operasi yang akan dilakukan (1:push, 2:pop, 3:print) : ");
        choice = masuk.nextInt();
        switch (choice) {
            case 1: {
                System.out.print("\nMasukkan huruf yang akan dipush : ");
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

    public static void main(String[] args) {
        stack_array_mod s = new stack_array_mod();
        do {
            s.menu();
        } while (s.keluar == 0);
    }
}
