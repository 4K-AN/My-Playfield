import java.util.Random;

/*
 * KELAS NODE (Tidak berubah dari program latihan)
 */
class Node {
    int data;
    Node nodeKiri;
    Node nodeKanan;

    public Node(int dt) {
        data = dt;
        nodeKiri = nodeKanan = null;
    }

    public void sisipDt(int dtSisip) {
        if (dtSisip < data) {
            if (nodeKiri == null)
                nodeKiri = new Node(dtSisip);
            else
                nodeKiri.sisipDt(dtSisip);
        } else if (dtSisip > data) {
            if (nodeKanan == null)
                nodeKanan = new Node(dtSisip);
            else
                nodeKanan.sisipDt(dtSisip);
        }
    }
}

/*
 * KELAS TREE (Dimodifikasi dengan jawaban latihan)
 */
public class Tree {
    private Node root;

    public Tree() {
        root = null;
    }

    public void sisipDtNode(int dtSisip) {
        if (root == null)
            root = new Node(dtSisip);
        else
            root.sisipDt(dtSisip);
    }

    // --- METODE TRAVERSAL (DARI MODUL ASLI) ---
    public void preorderTraversal() {
        preorder(root);
    }

    private void preorder(Node node) {
        if (node == null)
            return;
        System.out.printf("%d ", node.data);
        preorder(node.nodeKiri);
        preorder(node.nodeKanan);
    }

    public void inorderTraversal() {
        inorder(root);
    }

    private void inorder(Node node) {
        if (node == null)
            return;
        inorder(node.nodeKiri);
        System.out.printf("%d ", node.data);
        inorder(node.nodeKanan);
    }

    public void postorderTraversal() {
        postorder(root);
    }

    private void postorder(Node node) {
        if (node == null)
            return;
        postorder(node.nodeKiri);
        postorder(node.nodeKanan);
        System.out.printf("%d ", node.data);
    }

    // --- JAWABAN LATIHAN 9.9 ---

    /**
     * Latihan 1: Method untuk menghitung banyaknya node
     */
    public int countNodes() {
        return countNodesRec(root);
    }

    private int countNodesRec(Node node) {
        if (node == null) {
            return 0;
        }
        // 1 (node saat ini) + total node di kiri + total node di kanan
        return 1 + countNodesRec(node.nodeKiri) + countNodesRec(node.nodeKanan);
    }

    /**
     * Latihan 2: Method untuk menghitung banyaknya daun (leaf)
     */
    public int countLeaves() {
        return countLeavesRec(root);
    }

    private int countLeavesRec(Node node) {
        if (node == null) {
            return 0;
        }
        // Jika node tidak punya anak (kiri null DAN kanan null), 
        // maka ini adalah daun. Dihitung sebagai 1.
        if (node.nodeKiri == null && node.nodeKanan == null) {
            return 1;
        }
        // Jika bukan daun, jumlahkan total daun dari subtree kiri dan kanan
        return countLeavesRec(node.nodeKiri) + countLeavesRec(node.nodeKanan);
    }

    /**
     * Latihan 3: Method untuk menghitung tinggi pohon (Height)
     */
    public int getHeight() {
        // Konvensi:
        // - Tinggi tree kosong (root null) adalah -1
        // - Tinggi tree dengan 1 node (root saja) adalah 0
        return getHeightRec(root);
    }

    private int getHeightRec(Node node) {
        if (node == null) {
            return -1; // Basis untuk tree kosong
        }
        
        // Hitung tinggi subtree kiri dan kanan secara rekursif
        int leftHeight = getHeightRec(node.nodeKiri);
        int rightHeight = getHeightRec(node.nodeKanan);
        
        // Tinggi node ini adalah 1 + tinggi maksimum dari anak-anaknya
        return 1 + Math.max(leftHeight, rightHeight);
    }

    /**
     * Latihan 4: Method untuk menghitung panjang/jumlah level
     */
    public int getLevelCount() {
        // Jumlah level = Tinggi + 1
        // - Tree kosong (height -1) -> 0 level
        // - Tree 1 node (height 0) -> 1 level
        // - Tree 2 node (height 1) -> 2 level
        return getHeight() + 1;
    }


    // --- MAIN METHOD (DARI MODUL ASLI + MODIFIKASI LATIHAN) ---
    public static void main(String args[]) {
        Tree Tree = new Tree();
        int nilai;
        Random randomNumber = new Random();
        System.out.println("sisip nilai data berikut : ");

        // sisipDt 10 bilangan acak dari 0-99 ke dalam tree
        for (int i = 1; i <= 10; i++) {
            nilai = randomNumber.nextInt(100);
            System.out.print(nilai + " ");
            Tree.sisipDtNode(nilai);
        }

        // --- Bagian Traversal (Asli) ---
        System.out.println("\n\nPreorder traversal");
        Tree.preorderTraversal();
        System.out.println("\n\nInorder traversal");
        Tree.inorderTraversal();
        System.out.println("\n\nPostorder traversal");
        Tree.postorderTraversal();
        System.out.println();

        // --- PANGGIL METHOD LATIHAN 9.9 ---
        System.out.println("\n--- Analisis Latihan ---");

        // 1. Tampilkan hasil hitung node
        // (Catatan: Jika data duplikat, countNodes akan < 10)
        System.out.println("Banyaknya Node : " + Tree.countNodes());

        // 2. Tampilkan hasil hitung daun
        System.out.println("Banyaknya Daun : " + Tree.countLeaves());

        // 3. Tampilkan hasil tinggi pohon
        System.out.println("Tinggi Pohon   : " + Tree.getHeight());

        // 4. Tampilkan hasil panjang/level
        System.out.println("Jumlah Level   : " + Tree.getLevelCount());

        System.out.println();
    }
}