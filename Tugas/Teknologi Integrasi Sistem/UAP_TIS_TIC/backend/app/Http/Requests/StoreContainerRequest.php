<?php

namespace App\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;
use Illuminate\Contracts\Validation\Validator;
use Illuminate\Http\Exceptions\HttpResponseException;

class StoreContainerRequest extends FormRequest
{
    /**
     * Determine if the user is authorized to make this request.
     */
    public function authorize(): bool
    {
        return true;
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array<string, \Illuminate\Contracts\Validation\ValidationRule|array<mixed>|string>
     */
    public function rules(): array
    {
        $rules = [
            'container_code' => 'required|string|unique:containers,container_code|max:50',
            'type' => 'required|in:limbah_cair,limbah_padat,limbah_gas',
            'capacity' => 'required|numeric|min:1',
            'current_fill_level' => 'nullable|numeric|min:0',
            'location' => 'required|string|max:255',
            'status' => 'nullable|in:Active,Maintenance,Archived,Full',
            'last_maintenance_date' => 'nullable|date',
        ];

        // Conditional validation: jika status Full, current_fill_level harus >= 80% capacity
        if ($this->input('status') === 'Full') {
            $rules['current_fill_level'] = 'required|numeric|min:0';
        }

        return $rules;
    }

    /**
     * Get custom validation messages.
     */
    public function messages(): array
    {
        return [
            'container_code.required' => 'Kode kontainer wajib diisi.',
            'container_code.unique' => 'Kode kontainer sudah digunakan.',
            'type.required' => 'Tipe limbah wajib diisi.',
            'type.in' => 'Tipe limbah harus salah satu dari: limbah_cair, limbah_padat, limbah_gas.',
            'capacity.required' => 'Kapasitas kontainer wajib diisi.',
            'capacity.min' => 'Kapasitas minimal adalah 1.',
            'location.required' => 'Lokasi wajib diisi.',
            'status.in' => 'Status harus salah satu dari: Active, Maintenance, Archived, Full.',
        ];
    }

    /**
     * Handle a failed validation attempt.
     */
    protected function failedValidation(Validator $validator): void
    {
        throw new HttpResponseException(response()->json([
            'success' => false,
            'message' => 'Validasi gagal',
            'errors' => $validator->errors(),
        ], 422));
    }
}
