public class GraphLatihan {
    private class Node {
        private int data;
        private Node next;
        public Node(int dt, Node n) { data = dt; next = n; }
        public int getDt() { return data; }
        public Node getNext() { return next; }
    }

    private Node[] node;
    private int jNode;

    public GraphLatihan(int n) {
        jNode = n;
        node = new Node[jNode];
    }

    public void addAdj(int head, int adj) {
        node[head] = new Node(adj, node[head]);
    }

    // --- Jawaban Soal Latihan 1 ---
    public void cetakDegree(int targetNode) {
        // Out-degree: Hitung panjang linked list node tersebut
        int outDegree = 0;
        Node n = node[targetNode];
        while (n != null) {
            outDegree++;
            n = n.getNext();
        }

        // In-degree: Cari targetNode di list milik node lain
        int inDegree = 0;
        for (int i = 0; i < jNode; i++) {
            Node temp = node[i];
            while (temp != null) {
                if (temp.getDt() == targetNode) {
                    inDegree++;
                }
                temp = temp.getNext();
            }
        }
        
        System.out.println("Node " + targetNode + " info:");
        System.out.println(" - Out-degree (Keluar): " + outDegree);
        System.out.println(" - In-degree (Masuk)  : " + inDegree);
    }

    // --- Jawaban Soal Latihan 2 ---
    // Pada Directed Graph, jumlah tetangga sama dengan Out-degree
    public void cetakTetangga(int head) {
        System.out.print("Tetangga Node " + head + ": ");
        Node n = node[head];
        int count = 0;
        while(n != null){
            System.out.print(n.getDt() + " ");
            count++;
            n = n.getNext();
        }
        System.out.println("\nJumlah Tetangga: " + count);
    }

    public static void main(String[] args) {
        GraphLatihan g = new GraphLatihan(5);
        g.addAdj(0, 3); g.addAdj(0, 1);
        g.addAdj(1, 4); g.addAdj(1, 2);
        g.addAdj(2, 4); g.addAdj(2, 1);
        g.addAdj(4, 3);

        System.out.println("--- Hasil Latihan ---");
        g.cetakDegree(1); // Cek degree node 1
        System.out.println();
        g.cetakTetangga(0); // Cek tetangga node 0
    }
}