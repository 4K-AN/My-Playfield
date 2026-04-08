// NIM 245150707111012 , NAMA : AKHMAD SYAFIUL ANAM

<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;

class StudentController extends Controller
{
    
    public function index()
    {
        $students = [
            [
                "nim" => "245150707111012",
                "nama" => "Citra Dewi",
                "mataKuliah" => [
                    ["kode" => "CIE61205", "nama" => "PemWeb", "sks" => 3],
                    ["kode" => "COM60015", "nama" => "MatDis", "sks" => 2]
                ]
            ],
            [
                "nim" => "245150707111012",
                "nama" => "Andy Lau",
                "mataKuliah" => [
                    ["kode" => "CIE61205", "nama" => "PemWeb", "sks" => 3],
                    ["kode" => "CIE61206", "nama" => "JarKom", "sks" => 3],
                    ["kode" => "CIE61208", "nama" => "BasDat", "sks" => 3],
                ]
            ]
        ];

        return response()->json($students);
    }

    
    public function store(Request $request)
    {
        try {
            $validated = $request->validate([
                'nim' => 'required|digits:15',
                'nama' => 'required|string|max:50',
                'mataKuliah' => 'required|array',
                'mataKuliah.*.kode' => 'required|regex:/^[A-Z]{3}[0-9]{5}$/',
                'mataKuliah.*.nama' => 'required|string|max:50',
                'mataKuliah.*.sks' => 'required|numeric|min:1|max:6',
            ]);
        } catch (\Illuminate\Validation\ValidationException $th) {
            return response()->json([
                "message" => "Validation failed",
                "errors" => $th->validator->errors()
            ], 422);
        }

        return response()->json([
            "message" => "Student created successfully (dummy)",
            "data" => $validated
        ], 201);
    }

    
    public function update(Request $request, $nim)
    {
        try {
            $validated = $request->validate([
                'nama' => 'sometimes|required|string|max:50',
                'mataKuliah' => 'sometimes|required|array',
                'mataKuliah.*.kode' => 'sometimes|required|regex:/^[A-Z]{3}[0-9]{5}$/',
                'mataKuliah.*.nama' => 'sometimes|required|string|max:50',
                'mataKuliah.*.sks' => 'sometimes|required|numeric|min:1|max:6',
            ]);
        } catch (\Illuminate\Validation\ValidationException $th) {
            return response()->json([
                "message" => "Validation failed",
                "errors" => $th->validator->errors()
            ], 422);
        }

        return response()->json([
            "message" => "Student {$nim} updated successfully (dummy)",
            "data" => array_merge(['nim' => $nim], $validated)
        ]);
    }

    
    public function destroy($nim)
    {
        return response()->json([
            "message" => "Student {$nim} deleted successfully (dummy)"
        ]);
    }

    public function search(Request $request)
    {
    
        $nim = $request->query('nim');
        $nama = $request->query('nama');
        $kode_mk = $request->query('kode_mk');

       
        if (!$nim && !$nama && !$kode_mk) {
            return response()->json([
                "error" => "Parameter tidak ditemukan. Harap masukkan nim, nama, atau kode_mk."
            ], 400);
        }

    
        $students = [
            [
                "nim" => "245150707111012",
                "nama" => "Citra Dewi",
                "mataKuliah" => [
                    ["kode" => "CIE61205", "nama" => "PemWeb", "sks" => 3],
                    ["kode" => "COM60015", "nama" => "MatDis", "sks" => 2]
                ]
            ],
            [
                "nim" => "245150707111013",
                "nama" => "Andy Lau",
                "mataKuliah" => [
                    ["kode" => "CIE61205", "nama" => "PemWeb", "sks" => 3],
                    ["kode" => "CIE61206", "nama" => "JarKom", "sks" => 3],
                    ["kode" => "CIE61208", "nama" => "BasDat", "sks" => 3]
                ]
            ]
        ];

        $results = [];

      
        foreach ($students as $student) {
            $match = false;
            
           
            if ($nim && $student['nim'] == $nim) {
                $match = true;
            }
    
            elseif ($nama && stripos($student['nama'], $nama) !== false) {
                $match = true;
            }
          
            elseif ($kode_mk) {
                foreach ($student['mataKuliah'] as $mk) {
                    if ($mk['kode'] === $kode_mk) {
                        $match = true;
                        break;
                    }
                }
            }

        
            if ($match) {
                $results[] = $student;
            }
        }

        return response()->json([
            "message" => "Hasil pencarian",
            "data" => $results
        ]);
    }
}