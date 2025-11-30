import java.util.Arrays;

public class Tugas4_Dijkstra {
    int[][] matrix;
    int size;
    final int INF = 99999;

    public Tugas4_Dijkstra(int size) {
        this.size = size;
        matrix = new int[size][size];
        for (int i=0; i<size; i++) {
            Arrays.fill(matrix[i], INF);
            matrix[i][i] = 0;
        }
    }

    public void addEdge(int src, int dest, int weight) {
        matrix[src][dest] = weight;
        matrix[dest][src] = weight;
    }

    public void dijkstra(int start, int end) {
        int[] dist = new int[size];
        boolean[] visited = new boolean[size];
        Arrays.fill(dist, INF);
        dist[start] = 0;

        for (int i = 0; i < size; i++) {
            int u = -1, min = INF;
            for (int v = 0; v < size; v++) {
                if (!visited[v] && dist[v] <= min) {
                    min = dist[v];
                    u = v;
                }
            }
            if (u == -1 || dist[u] == INF) break;
            visited[u] = true;

            for (int v = 0; v < size; v++) {
                if (!visited[v] && matrix[u][v] != INF && dist[u] + matrix[u][v] < dist[v]) {
                    dist[v] = dist[u] + matrix[u][v];
                }
            }
        }
        System.out.println("Jarak terpendek node " + start + " ke " + end + " = " + dist[end]);
    }

    public static void main(String[] args) {
        Tugas4_Dijkstra g = new Tugas4_Dijkstra(5);
        g.addEdge(0, 1, 10);
        g.addEdge(0, 4, 100);
        g.addEdge(1, 4, 10);
        
        g.dijkstra(0, 4); // Jalur 0->1->4 (20) lebih cepat dari 0->4 (100)
    }
}