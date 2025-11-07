import java.util.ArrayList;


public class CompleteBinaryTreeArray {


    private ArrayList<Integer> tree;

    public CompleteBinaryTreeArray() {
        tree = new ArrayList<>();
    }


    public void add(int data) {
        tree.add(data);
        System.out.println("Menambahkan " + data + " di indeks " + (tree.size() - 1));
    }


    public int getParentIndex(int index) {
        if (index == 0) {
            return -1;
        }
        return (index - 1) / 2;
    }

    public int getLeftChildIndex(int index) {
        return (2 * index) + 1;
    }

    public int getRightChildIndex(int index) {
        return (2 * index) + 2;
    }

    public void printLevelOrder() {
        if (tree.isEmpty()) {
            System.out.println("Pohon kosong.");
            return;
        }

        System.out.println("\n--- Complete Binary Tree (Level Order) ---");
        int size = tree.size();
        for (int i = 0; i < size; i++) {
            System.out.println("Node di Indeks " + i + ": " + tree.get(i));
        }
        System.out.println("-------------------------------------------");
    }

    
    public void printAllRelations() {
        if (tree.isEmpty()) return;

        System.out.println("\n--- Relasi Parent-Child ---");
        for (int i = 0; i < tree.size(); i++) {
            System.out.print("Node " + tree.get(i) + " (di idx " + i + "): ");


            int parentIdx = getParentIndex(i);
            if (parentIdx != -1) {
                System.out.print("Parent = " + tree.get(parentIdx) + ". ");
            } else {
                System.out.print("Ini adalah ROOT. ");
            }

    
            int leftIdx = getLeftChildIndex(i);
            if (leftIdx < tree.size()) {
                System.out.print("Anak Kiri = " + tree.get(leftIdx) + ". ");
            }

        
            int rightIdx = getRightChildIndex(i);
            if (rightIdx < tree.size()) {
                System.out.print("Anak Kanan = " + tree.get(rightIdx) + ". ");
            }
            System.out.println();
        }
    }


    public static void main(String[] args) {
        CompleteBinaryTreeArray cbt = new CompleteBinaryTreeArray();

    
        cbt.add(50);
        cbt.add(30);
        cbt.add(70);
        cbt.add(20);
        cbt.add(40);
        cbt.add(60);
        
    

        cbt.printLevelOrder();
        cbt.printAllRelations();
    }
}
