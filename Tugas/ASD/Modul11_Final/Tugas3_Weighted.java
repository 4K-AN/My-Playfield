import java.util.Arrays;

public class Tugas3_Weighted {
    int[][] matrix;
    int size;
    final int INF = 99999;

    public Tugas3_Weighted(int size) {
        this.size = size;
        matrix = new int[size][size];
        for (int[] row : matrix) Arrays.fill(row, INF);
        for (int i=0; i<size; i++) matrix[i][i] = 0;
    }

    public void addEdge(int src, int dest, int weight) {
        matrix[src][dest] = weight;
        matrix[dest][src] = weight;
    }

    public void print() {
        System.out.println("Weighted Matrix:");
        for(int[] row : matrix) {
            System.out.println(Arrays.toString(row));
        }
    }

    public static void main(String[] args) {
        Tugas3_Weighted g = new Tugas3_Weighted(3);
        g.addEdge(0, 1, 50);
        g.addEdge(1, 2, 25);
        g.print();
    }
}