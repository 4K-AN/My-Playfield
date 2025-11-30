public class Tugas1_Matrix {
    int[][] matrix;
    int size;

    public Tugas1_Matrix(int size) {
        this.size = size;
        matrix = new int[size][size];
    }

    public void addEdge(int src, int dest) {
        matrix[src][dest] = 1;
        matrix[dest][src] = 1; // Graf tak berarah
    }

    public void printGraph() {
        System.out.println("Adjacency Matrix:");
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                System.out.print(matrix[i][j] + " ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        Tugas1_Matrix g = new Tugas1_Matrix(4);
        g.addEdge(0, 1);
        g.addEdge(0, 2);
        g.addEdge(1, 3);
        g.printGraph();
    }
}