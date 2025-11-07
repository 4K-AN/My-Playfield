
class NodeCSLL {

    Object data;
    NodeCSLL setelah;
}

public class CircularSingleLinkedListNoJumlah {

    private NodeCSLL pAwal, pAkhir;

    public CircularSingleLinkedListNoJumlah() {
        pAwal = null;
        pAkhir = null;
    }

    public void SisipDataDiAwal(Object data) {
        NodeCSLL pBaru = new NodeCSLL();
        pBaru.data = data;
        pBaru.setelah = pBaru;

        if (pAwal == null) {
            pAwal = pBaru;
            pAkhir = pBaru;
        } else {
            pBaru.setelah = pAwal;
            pAkhir.setelah = pBaru;
            pAwal = pBaru;
        }
    }

    public void SisipDataDiAkhir(Object data) {
        NodeCSLL pBaru = new NodeCSLL();
        pBaru.data = data;
        pBaru.setelah = pBaru;

        if (pAwal == null) {
            pAwal = pBaru;
            pAkhir = pBaru;
        } else {
            pBaru.setelah = pAwal;
            pAkhir.setelah = pBaru;
            pAkhir = pBaru;
        }
    }

    public void hapusData(Object dtHapus) {
        if (pAwal != null) {
            NodeCSLL pSbl, pKini, pHapus;
            pSbl = null;
            pKini = pAwal;
            boolean ketemu = false;

            // Loop sampai kembali ke pAwal atau data ditemukan
            do {
                if (pKini.data.equals(dtHapus)) {
                    ketemu = true;
                    break;
                } else {
                    pSbl = pKini;
                    pKini = pKini.setelah;
                }
            } while (pKini != pAwal);

            if (ketemu) {
                if (pAwal == pAkhir) {
                    // Hanya ada satu node
                    pHapus = pAwal;
                    pAwal = null;
                    pAkhir = null;
                    pHapus = null;
                } else if (pSbl == null) {
                    // Hapus node awal
                    pHapus = pAwal;
                    pAwal = pKini.setelah;
                    pAkhir.setelah = pAwal;
                    pHapus = null;
                } else {
                    // Hapus node di tengah atau akhir
                    if (pAkhir == pKini) {
                        pAkhir = pSbl;
                    }
                    pSbl.setelah = pKini.setelah;
                    pHapus = pKini;
                    pHapus = null;
                }
            }
        }
    }

    public void hapusSatuDataDiAwal() {
        if (pAwal != null) {
            NodeCSLL pHapus;

            if (pAwal == pAkhir) {
                pHapus = pAwal;
                pAwal = null;
                pAkhir = null;
                pHapus = null;
            } else {
                pHapus = pAwal;
                pAwal = pAwal.setelah;
                pAkhir.setelah = pAwal;
                pHapus = null;
            }
        }
    }

    public void hapusSatuDataDiAkhir() {
        if (pAwal != null) {
            NodeCSLL pHapus, pBantu;

            if (pAwal == pAkhir) {
                pHapus = pAwal;
                pAwal = null;
                pAkhir = null;
                pHapus = null;
            } else {
                pBantu = pAwal;
                while (pBantu.setelah != pAkhir) {
                    pBantu = pBantu.setelah;
                }
                pHapus = pAkhir;
                pAkhir = pBantu;
                pAkhir.setelah = pAwal;
                pHapus = null;
            }
        }
    }

    public void cetak(String Komentar) {
        System.out.println(Komentar);
        if (pAwal != null) {
            NodeCSLL pCetak = pAwal;
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
        CircularSingleLinkedListNoJumlah csll = new CircularSingleLinkedListNoJumlah();

        System.out.println("=== PENGUJIAN CSLL TANPA VARIABEL JUMLAH ===\n");

        csll.SisipDataDiAwal(new Integer(50));
        csll.SisipDataDiAwal(new Integer(60));
        csll.SisipDataDiAwal(new Integer(70));
        csll.cetak("Data awal");

        csll.SisipDataDiAkhir(new Integer(100));
        csll.cetak("Setelah sisip 100 di akhir");

        csll.hapusData(60);
        csll.cetak("Setelah hapus 60");

        csll.hapusSatuDataDiAwal();
        csll.cetak("Setelah hapus data di awal");

        csll.hapusSatuDataDiAkhir();
        csll.cetak("Setelah hapus data di akhir");
    }
}
