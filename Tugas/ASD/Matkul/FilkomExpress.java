
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class FilkomExpress {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        int n = scanner.nextInt();
        scanner.nextLine();

        if (n < 1 || n > 100) {
            System.out.println("Error: jumlah paket harus antara 1 hingga 100");
            return;
        }

        String[] weightsStr = scanner.nextLine().split(" ");

        Integer[] weights = new Integer[n];

        for (int i = 0; i < n; i++) {

            weights[i] = Integer.parseInt(weightsStr[i]);
            if (weights[i] < 1 || weights[i] > 20) {
                System.out.println("Error: semua berat paket harus antara 1 hingga 20");
                return;
            }
        }

        double sum = 0;

        for (Integer weight : weights) {

            sum += weight;
        }
        double average = sum / n;
        System.out.printf("Rata-rata: %.2f%n", average);

        Integer max = weights[0];
        Integer min = weights[0];
        for (Integer weight : weights) {
            if (weight > max) {
                max = weight;
            }
            if (weight < min) {
                min = weight;
            }
        }
        System.out.println("Maksimum: " + max);
        System.out.println("Minimum: " + min);

        List<Integer> heavierThanAverage = new ArrayList<>();
        for (Integer weight : weights) {
            if (weight > average) {
                heavierThanAverage.add(weight);
            }
        }
        Collections.sort(heavierThanAverage);
        System.out.println("Lebih dari rata-rata: " + heavierThanAverage);

        List<Integer> primeWeights = new ArrayList<>();
        for (Integer weight : weights) {
            if (isPrime(weight)) {
                primeWeights.add(weight);
            }
        }
        Collections.sort(primeWeights);
        System.out.println("Bilangan prima: " + primeWeights);
    }

    public static boolean isPrime(Integer num) {
        if (num <= 1) {
            return false;
        }

        for (int i = 2; i <= Math.sqrt(num); i++) {
            if (num % i == 0) {
                return false;
            }
        }
        return true;
    }
}
