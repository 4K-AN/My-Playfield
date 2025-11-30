import java.util.*;

public class Main {
    static Map<String, List<String>> graph = new HashMap<>();
    
    static void transfer(String pengirim, String penerima) {
        if (!graph.containsKey(pengirim)) {
            graph.put(pengirim, new ArrayList<>());
        }
        if (!graph.containsKey(penerima)) {
            graph.put(penerima, new ArrayList<>());
        }
        graph.get(pengirim).add(penerima);
    }
    
    static void cekCuciUang(String nama) {
        if (!graph.containsKey(nama)) {
            System.out.println("Nama " + nama + " tidak ditemukan");
            return;
        }
        
        Set<String> visited = new HashSet<>();
        Set<String> recStack = new HashSet<>();
        
        if (dfsCycle(nama, visited, recStack)) {
            System.out.println(nama + " TERINDIKASI CUCI UANG");
        } else {
            System.out.println(nama + " Transaksi Normal");
        }
    }
    
    static boolean dfsCycle(String node, Set<String> visited, Set<String> recStack) {
        visited.add(node);
        recStack.add(node);
        
        if (graph.containsKey(node)) {
            for (String neighbor : graph.get(node)) {
                if (!visited.contains(neighbor)) {
                    if (dfsCycle(neighbor, visited, recStack)) {
                        return true;
                    }
                } else if (recStack.contains(neighbor)) {
                    return true;
                }
            }
        }
        
        recStack.remove(node);
        return false;
    }
    
    static void cekJarak(String orang1, String orang2) {
        if (!graph.containsKey(orang1)) {
            System.out.println("Nama " + orang1 + " tidak ditemukan");
            return;
        }
        if (!graph.containsKey(orang2)) {
            System.out.println("Nama " + orang2 + " tidak ditemukan");
            return;
        }
        
        Queue<String> queue = new LinkedList<>();
        Map<String, Integer> distance = new HashMap<>();
        
        queue.offer(orang1);
        distance.put(orang1, 0);
        
        while (!queue.isEmpty()) {
            String current = queue.poll();
            
            if (current.equals(orang2)) {
                System.out.println("Jarak " + orang1 + " ke " + orang2 + ": " + distance.get(current) + " transfer");
                return;
            }
            
            if (graph.containsKey(current)) {
                for (String neighbor : graph.get(current)) {
                    if (!distance.containsKey(neighbor)) {
                        distance.put(neighbor, distance.get(current) + 1);
                        queue.offer(neighbor);
                    }
                }
            }
        }
        
        System.out.println(orang1 + " tidak memiliki hubungan dengan " + orang2);
    }
    
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        
        int n = sc.nextInt();
        sc.nextLine();
        
        for (int i = 0; i < n; i++) {
            String line = sc.nextLine().trim();
            String[] parts = line.split("\\s+");
            String cmd = parts[0].toUpperCase();
            
            if (cmd.equals("TRANSFER")) {
                if (parts.length >= 3) {
                    transfer(parts[1], parts[2]);
                }
            } else if (cmd.equals("CEK_CUCI_UANG")) {
                if (parts.length >= 2) {
                    cekCuciUang(parts[1]);
                }
            } else if (cmd.equals("CEK_JARAK")) {
                if (parts.length >= 3) {
                    cekJarak(parts[1], parts[2]);
                }
            }
        }
        
        sc.close();
    }
}