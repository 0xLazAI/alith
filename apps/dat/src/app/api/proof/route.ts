// api/files/route.ts
import { NextResponse, type NextRequest } from 'next/server'
import axios, { AxiosResponse } from 'axios'

export async function POST(request: NextRequest) {
  try {
    const { job_id, file_id, file_url, node_url, encryption_key, encryption_seed, nonce, proof_url } =
      await request.json()
    const proofRequest = {
      job_id: job_id,
      file_id: file_id,
      file_url: file_url,
      encryption_key: encryption_key,
      encryption_seed: encryption_seed,
      nonce: nonce,
      proof_url: proof_url,
    }
    const response: AxiosResponse = await axios.post(`${node_url}/proof`, proofRequest, {
      headers: { 'Content-Type': 'application/json' },
    })
    if (response.status != 200) {
      throw Error(response.data)
    }
    return NextResponse.json(response.data, { status: 200 })
  } catch (e) {
    console.log(e)
    return NextResponse.json({ error: e }, { status: 500 })
  }
}
