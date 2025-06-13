// api/files/route.ts
import { NextResponse, type NextRequest } from 'next/server'
import { PinataIPFS } from 'alith/data/storage'
import { Buffer } from 'buffer'

export async function POST(request: NextRequest) {
  try {
    const { name, content } = await request.json()
    const dataBuffer = Buffer.from(content, 'base64')
    const ipfs = new PinataIPFS()
    const fileMeta = await ipfs.upload({
      name: name,
      data: dataBuffer,
      token: process.env.PINATA_JWT || '',
    })
    const url = await ipfs.getShareLink({ token: process.env.PINATA_JWT || '', id: fileMeta.id })
    return NextResponse.json(url, { status: 200 })
  } catch (e) {
    console.log(e)
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 })
  }
}
