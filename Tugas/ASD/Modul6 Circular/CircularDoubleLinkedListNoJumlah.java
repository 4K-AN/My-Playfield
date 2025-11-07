
class NodeCDLL {

    Object data;
    NodeCDLL sebelum;
    NodeCDLL setelah;
}

public class CircularDoubleLinkedListNoJumlah {

    private NodeCDLL pAwal, pAkhir;

    public CircularDoubleLinkedListNoJumlah() {
        pAwal = null;
        pAkhir = null;
    }

    public void SisipDataDiAwal(Object data) {
        NodeCDLL pBaru = new NodeCDLL();
        pBaru.data = data;
        pBaru.sebelum = pBaru;
        pBaru.setelah = pBaru;

        if (pAwal == null) {
            pAwal = pBaru;
            pAkhir = pBaru;
        } else {
            pBaru.sebelum = pAkhir;
            pBaru.setelah = pAwal;
            pAwal.sebelum = pBaru;
            pAkhir.setelah = pBaru;
            pAwal = pBaru;
        }
    }

    public void SisipDataDiAkhir(Object data) {
        NodeCDLL pBaru = new NodeCDLL();
        pBaru.data = data;
        pBaru.sebelum = pBaru;
        pBaru.setelah = pBaru;

        if (pAwal == null) {
            pAwal = pBaru;
            pAkhir = pBaru;
        } else {
            pBaru.sebelum = pAkhir;
            pBaru.setelah = pAwal;
            pAkhir.setelah = pBaru;
            pAwal.sebelum = pBaru;
            pAkhir = pBaru;
        }
    }

    public void hapusData(Object dtHapus) {
        if (pAwal != null) {
            NodeCDLL pKini, pHapus;
            pKini = pAwal;
            boolean ketemu = false;

            do {
                if (pKini.data.equals(dtHapus)) {
                    ketemu = true;
                    break;
                } else {
                    pKini = pKini.setelah;
                }
            } while (pKini != pAwal);

            if (ketemu) {
                if (pAwal == pAkhir) {
                    pHapus = pAwal;
                    pAwal = null;
                    pAkhir = null;
                    pHapus = null;
                } else if (pKini == pAwal) {
                    pHapus = pAwal;
                    pAwal = pAwal.setelah;
                    pAwal.sebelum = pAkhir;
                    pAkhir.setelah = pAwal;
                    pHapus = null;
                } else if (pKini == pAkhir) {
                    pHapus = pAkhir;
                    pAkhir = pAkhir.sebelum;
                    pAkhir.setelah = pAwal;
                    pAwal.sebelum = pAkhir;
                    pHapus = null;
                } else {
                    pHapus = pKini;
                    pKini.sebelum.setelah = pKini.setelah;
                    pKini.setelah.sebelum = pKini.sebelum;
                    pHapus = null;
                }
            }
        }
    }

    public void cetak(String Komentar) {
        System.out.println(Komentar);
        if (pAwal != null) {
            NodeCDLL pCetak = pAwal;
            do {
                System.out.print(pCetak.data + "->");
                pCetak = pCetak.setelah;
            } while (pCetak != pAwal);
            System.out.println();
        } else {
            System.out.println("List Kosong");
        }
    }

    public static void main(String[] args) {
        CircularDoubleLinkedListNoJumlah cdll = new CircularDoubleLinkedListNoJumlah();

        System.out.println("=== PENGUJIAN CDLL TANPA VARIABEL JUMLAH ===\n");

        cdll.SisipDataDiAwal(new Integer(50));
        cdll.SisipDataDiAwal(new Integer(60));
        cdll.SisipDataDiAwal(new Integer(70));
        cdll.cetak("Data awal");

        cdll.SisipDataDiAkhir(new Integer(100));
        cdll.cetak("Setelah sisip 100 di akhir");

        cdll.hapusData(60);
        cdll.cetak("Setelah hapus 60");

        cdll.hapusData(70);
        cdll.cetak("Setelah hapus 70");
    }
}
