'use client'

import { useState, useEffect } from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import {
  faWallet,
  faFile,
  faCloudUploadAlt,
  faTimes,
  faLink,
  faCheck,
  faSpinner,
  faLock,
  faArrowLeft,
  faRefresh,
  faExclamationCircle,
} from '@fortawesome/free-solid-svg-icons'
import { faEthereum } from '@fortawesome/free-brands-svg-icons'
import { BrowserProvider, ethers, EventLog } from 'ethers'
import { DATA_REGISTRY_CONTRACT_ABI, VERIFIED_COMPUTING_CONTRACT_ABI, ContractConfig } from 'alith/lazai'
import { encrypt } from 'alith/data/crypto'
import NodeRSA from 'node-rsa'

const ENCRYPTION_SEED = 'Sign to retrieve your encryption key'

export default function Home() {
  // State variables
  const [currentStep, setCurrentStep] = useState(1)
  const [walletConnected, setWalletConnected] = useState(false)
  const [walletAddress, setWalletAddress] = useState('')
  const [password, setPassword] = useState('')
  const [dataName, setDataName] = useState('')
  const [privacyData, setPrivacyData] = useState('')
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadComplete, setUploadComplete] = useState(false)
  const [ipfsUrl, setIpfsUrl] = useState('')
  const [datGenerated, setDatGenerated] = useState(false)
  const [fileId, setFileId] = useState('')
  const [jobId, setJobId] = useState('')
  const [nodeAddress, setNodeAddress] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [notification, setNotification] = useState({
    show: false,
    type: 'success',
    message: '',
  })

  // Check Metamask is installed
  useEffect(() => {
    if (typeof window.ethereum === 'undefined') {
      showNotification('error', 'Metamask not detected. Please install Metamask to continue.')
    }
  }, [])

  // Sign the message use the user address and message
  const signMessage = async (address: string, message: string): Promise<string> => {
    if (!window.ethereum) {
      throw new Error('MetaMask not found')
    }

    const provider = new BrowserProvider(window.ethereum)
    const signer = await provider.getSigner(address)
    const signature = await signer.signMessage(message)
    return signature
  }

  const uint8ArrayToBase64 = (array: Uint8Array): string => {
    let result = ''
    const chunkSize = 1024
    for (let i = 0; i < array.length; i += chunkSize) {
      const chunk = array.subarray(i, i + chunkSize)
      result += btoa(String.fromCharCode.apply(null, Array.from(chunk)))
    }
    return result
  }

  // Connect wallet
  const connectWallet = async () => {
    try {
      if (!window.ethereum) {
        showNotification('error', 'Please install Metamask to continue.')
        return
      }

      setLoading(true)
      const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' })
      setWalletAddress(accounts[0])
      setWalletConnected(true)
      setLoading(false)
      showNotification('success', 'Wallet connected successfully!')
      setTimeout(() => {
        setCurrentStep(2)
      }, 1000)
    } catch (error) {
      console.error('Error connecting to Metamask:', error)
      setLoading(false)
      showNotification('error', 'Failed to connect to Metamask. Please try again.')
    }
  }

  const encryptAndUploadData = async () => {
    if (!dataName || !privacyData) {
      showNotification('error', 'Please enter both file name and privacy data.')
      return
    }

    setLoading(true)
    setCurrentStep(3)
    setUploadProgress(0)

    try {
      updateProgress(20, 'Generating encryption key...')
      const password = await signMessage(walletAddress, ENCRYPTION_SEED)
      setPassword(password)

      updateProgress(40, 'Encrypting your data...')
      const encryptedData = await encrypt(Uint8Array.from(privacyData), password)

      updateProgress(60, 'Uploading to IPFS...')
      const uploadRequest = await fetch('/api/files', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: dataName,
          content: uint8ArrayToBase64(encryptedData),
        }),
      })
      updateProgress(80, 'Finalizing upload...')
      const url = await uploadRequest.json()

      setIpfsUrl(url)
      setUploadProgress(100)
      setUploadComplete(true)
      setLoading(false)
      showNotification('success', 'Data encrypted and uploaded to IPFS successfully!')
    } catch (error) {
      console.error('Error encrypting and uploading data:', error)
      setLoading(false)
      showNotification('error', 'Failed to upload data. Please try again.')
    }
  }

  const generateDAT = async () => {
    setLoading(true)
    setCurrentStep(4)

    try {
      const provider = new ethers.BrowserProvider(window.ethereum)
      const signer = await provider.getSigner()
      const config = new ContractConfig()
      const registryContract = new ethers.Contract(config.dataRegistryAddress, DATA_REGISTRY_CONTRACT_ABI, signer)
      const vcContract = new ethers.Contract(config.verifiedComputingAddress, VERIFIED_COMPUTING_CONTRACT_ABI, signer)
      updateProgress(20, 'Registering file on blockchain...')
      const url = ipfsUrl
      let fileId = await registryContract.getFileIdByUrl(url)
      if (fileId == BigInt(0)) {
        const tx = await registryContract.addFile(url)
        const receipt = await tx.wait()
        fileId = await registryContract.getFileIdByUrl(url)
      }
      console.log('file id:', fileId)
      setFileId(fileId)

      updateProgress(40, 'Requesting verification proof...')
      const tx = await vcContract.requestProof(fileId, { value: 100 })
      const receipt = await tx.wait()
      const jobIds = await vcContract.fileJobIds(fileId)
      const jobId = jobIds[jobIds.length - 1]
      const job = await vcContract.getJob(jobId)
      console.log('job id:', jobId)
      setJobId(jobId.toString())

      updateProgress(60, 'Processing verification...')
      const nodeInfo = await vcContract.getNode(job.nodeAddress)
      const nodeUrl = nodeInfo.url
      const pubKey = nodeInfo.publicKey
      setNodeAddress(job.nodeAddress)
      const rsa = new NodeRSA(pubKey, 'pkcs1-public-pem')
      const encryptedKey = rsa.encrypt(password, 'hex')
      const proofRequest = {
        job_id: Number(jobId),
        file_id: Number(fileId),
        file_url: url,
        node_url: nodeUrl,
        encryption_key: encryptedKey,
        encryption_seed: ENCRYPTION_SEED,
        nonce: null,
        proof_url: null,
      }
      updateProgress(80, 'Request Proof from the verified computing node...')
      const response = await fetch('/api/proof', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(proofRequest),
      })

      if (response.status === 200) {
        console.log('Proof request sent successfully:', response.json())
        showNotification('success', 'Proof request sent successfully')
      } else {
        console.log('Failed to send proof request:', response.json())
        throw Error('Failed to send proof request: ' + response.json())
      }

      updateProgress(80, 'Generating DAT...')
      const _tx = await registryContract.requestReward(fileId, 1)
      const _receipt = await tx.wait()

      setUploadProgress(100)
      setDatGenerated(true)
      setLoading(false)
      showNotification('success', 'DAT generated successfully!')
    } catch (error) {
      console.error('Error generating DAT:', error)
      setLoading(false)
      showNotification('error', 'Failed to generate DAT. Please try again.')
    }
  }

  const resetApp = () => {
    setCurrentStep(1)
    setDataName('')
    setPrivacyData('')
    setUploadProgress(0)
    setUploadComplete(false)
    setDatGenerated(false)
    setFileId('')
    setJobId('')
    setNodeAddress('')
    setError('')
  }

  const updateProgress = (percent, message) => {
    setUploadProgress(percent)
  }

  const simulateDelay = (ms) => {
    return new Promise((resolve) => setTimeout(resolve, ms))
  }

  const shortenAddress = (address) => {
    if (!address) return ''
    return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`
  }

  const showNotification = (type, message) => {
    setNotification({
      show: true,
      type,
      message,
    })

    setTimeout(() => {
      setNotification({
        show: false,
        type: 'success',
        message: '',
      })
    }, 3000)
  }

  return (
    <div className="grid grid-rows-[auto_1fr_auto] min-h-screen bg-gradient-to-br from-gray-900 to-black text-white font-sans">
      {/* Navigation Bar */}
      <header className="py-4 px-6 border-b border-gray-800 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <FontAwesomeIcon icon={faLock} className="text-primary text-xl" />
          <h1 className="text-xl font-bold">LazAI DAT Demo</h1>
        </div>

        {walletConnected ? (
          <div className="flex items-center gap-3">
            <FontAwesomeIcon icon={faEthereum} className="text-gray-400" />
            <span className="text-sm text-gray-300">{shortenAddress(walletAddress)}</span>
          </div>
        ) : (
          <button
            onClick={connectWallet}
            className="px-4 py-2 rounded-lg bg-primary hover:bg-primary/90 text-white transition-all duration-300 flex items-center gap-2"
            disabled={loading}
          >
            {loading ? (
              <FontAwesomeIcon icon={faSpinner} spin className="mr-2" />
            ) : (
              <FontAwesomeIcon icon={faWallet} className="mr-2" />
            )}
            Connect Wallet
          </button>
        )}
      </header>

      {/* Main content */}
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Steps */}
        <div className="mb-10">
          <div className="flex justify-between">
            <div className={`flex flex-col items-center ${currentStep >= 1 ? 'text-primary' : 'text-gray-500'}`}>
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 ${currentStep >= 1 ? 'bg-primary/20' : 'bg-gray-800'}`}
              >
                {currentStep >= 1 ? <FontAwesomeIcon icon={faCheck} /> : <span className="font-bold">1</span>}
              </div>
              <span className="text-sm font-medium">Connect Wallet</span>
            </div>

            <div className="flex-1 flex items-center mx-2">
              <div className={`h-1 flex-1 ${currentStep >= 2 ? 'bg-primary' : 'bg-gray-800'}`} />
            </div>

            <div className={`flex flex-col items-center ${currentStep >= 2 ? 'text-primary' : 'text-gray-500'}`}>
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 ${currentStep >= 2 ? 'bg-primary/20' : 'bg-gray-800'}`}
              >
                {currentStep >= 2 ? <FontAwesomeIcon icon={faCheck} /> : <span className="font-bold">2</span>}
              </div>
              <span className="text-sm font-medium">Enter Data</span>
            </div>

            <div className="flex-1 flex items-center mx-2">
              <div className={`h-1 flex-1 ${currentStep >= 3 ? 'bg-primary' : 'bg-gray-800'}`} />
            </div>

            <div className={`flex flex-col items-center ${currentStep >= 3 ? 'text-primary' : 'text-gray-500'}`}>
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 ${currentStep >= 3 ? 'bg-primary/20' : 'bg-gray-800'}`}
              >
                {currentStep >= 3 ? <FontAwesomeIcon icon={faCheck} /> : <span className="font-bold">3</span>}
              </div>
              <span className="text-sm font-medium">Upload to IPFS</span>
            </div>

            <div className="flex-1 flex items-center mx-2">
              <div className={`h-1 flex-1 ${currentStep >= 4 ? 'bg-primary' : 'bg-gray-800'}`} />
            </div>

            <div className={`flex flex-col items-center ${currentStep >= 4 ? 'text-primary' : 'text-gray-500'}`}>
              <div
                className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 ${currentStep >= 4 ? 'bg-primary/20' : 'bg-gray-800'}`}
              >
                {currentStep >= 4 ? <FontAwesomeIcon icon={faCheck} /> : <span className="font-bold">4</span>}
              </div>
              <span className="text-sm font-medium">Generate DAT</span>
            </div>
          </div>
        </div>

        {}
        <div className="bg-gray-800/50 rounded-2xl p-6 md:p-10 border border-gray-700 shadow-xl">
          {/* Step 1: Connect wallet */}
          {currentStep === 1 && (
            <div className="flex flex-col items-center text-center">
              <div className="w-20 h-20 rounded-full bg-primary/20 flex items-center justify-center mb-6">
                <FontAwesomeIcon icon={faWallet} className="text-3xl text-primary" />
              </div>

              <h2 className="text-2xl font-bold mb-4">Connect Your Wallet</h2>
              <p className="text-gray-400 mb-8 max-w-md">
                Connect your Metamask wallet to securely sign transactions and manage your privacy data.
              </p>

              <button
                onClick={connectWallet}
                className="px-8 py-4 rounded-xl bg-primary hover:bg-primary/90 text-white transition-all duration-300 flex items-center gap-3 shadow-lg shadow-primary/20"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <FontAwesomeIcon icon={faSpinner} spin />
                    <span>Connecting...</span>
                  </>
                ) : (
                  <>
                    <FontAwesomeIcon icon={faWallet} />
                    <span>Connect Metamask</span>
                  </>
                )}
              </button>
            </div>
          )}

          {/* Step 2: Input Data */}
          {currentStep === 2 && (
            <div>
              <h2 className="text-2xl font-bold mb-6">Enter Your Privacy Data</h2>
              <p className="text-gray-400 mb-8">This data will be encrypted and stored securely on IPFS.</p>

              <div className="space-y-6">
                <div>
                  <label htmlFor="data-name" className="block text-sm font-medium mb-2">
                    File Name
                  </label>
                  <input
                    type="text"
                    id="data-name"
                    placeholder="Enter file name"
                    value={dataName}
                    onChange={(e) => setDataName(e.target.value)}
                    className="w-full px-4 py-3 rounded-lg bg-gray-900 border border-gray-700 focus:border-primary focus:ring-2 focus:ring-primary/30 outline-none transition-all duration-300"
                  />
                </div>

                <div>
                  <label htmlFor="privacy-data" className="block text-sm font-medium mb-2">
                    Privacy Data
                  </label>
                  <textarea
                    id="privacy-data"
                    rows="8"
                    placeholder="Enter your privacy data here..."
                    value={privacyData}
                    onChange={(e) => setPrivacyData(e.target.value)}
                    className="w-full px-4 py-3 rounded-lg bg-gray-900 border border-gray-700 focus:border-primary focus:ring-2 focus:ring-primary/30 outline-none transition-all duration-300"
                  />
                </div>
              </div>

              <div className="mt-8 flex justify-between">
                <button
                  onClick={() => setCurrentStep(1)}
                  className="px-6 py-3 rounded-lg bg-gray-700 hover:bg-gray-600 text-white transition-all duration-300 flex items-center gap-2"
                >
                  <FontAwesomeIcon icon={faArrowLeft} />
                  <span>Back</span>
                </button>

                <button
                  onClick={encryptAndUploadData}
                  className="px-8 py-3 rounded-lg bg-primary hover:bg-primary/90 text-white transition-all duration-300 flex items-center gap-2 shadow-lg shadow-primary/20"
                  disabled={loading || !dataName || !privacyData}
                >
                  {loading ? (
                    <>
                      <FontAwesomeIcon icon={faSpinner} spin />
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <FontAwesomeIcon icon={faLock} />
                      <span>Encrypt & Upload</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Step 3: Upload data to IPFS */}
          {currentStep === 3 && (
            <div>
              <h2 className="text-2xl font-bold mb-6">Uploading to IPFS</h2>
              <p className="text-gray-400 mb-8">Your data is being encrypted and uploaded to the IPFS network.</p>

              <div className="mb-8">
                <div className="w-full bg-gray-700 rounded-full h-2.5">
                  <div className="bg-primary h-2.5 rounded-full" style={{ width: `${uploadProgress}%` }} />
                </div>
                <p className="text-sm text-gray-400 mt-2">{uploadProgress}% complete</p>
              </div>

              {uploadComplete && (
                <div className="mb-8 p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center">
                      <FontAwesomeIcon icon={faCheck} className="text-green-500" />
                    </div>
                    <div>
                      <h3 className="font-medium">Upload Successful!</h3>
                      <p className="text-sm text-gray-400">Your data has been encrypted and uploaded to IPFS.</p>
                    </div>
                  </div>

                  <div className="mt-4 p-4 rounded-lg bg-gray-900 border border-gray-700">
                    <div className="flex items-start gap-3">
                      <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center mt-1">
                        <FontAwesomeIcon icon={faFile} className="text-primary" />
                      </div>
                      <div className="flex-1">
                        <h3 className="font-medium">{dataName}</h3>
                        <p className="text-sm text-gray-400 truncate">{ipfsUrl}</p>
                        <a
                          href={ipfsUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm text-primary hover:underline flex items-center mt-1"
                        >
                          <FontAwesomeIcon icon={faLink} className="mr-1" />
                          <span>Download</span>
                        </a>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="mt-8 flex justify-between">
                <button
                  onClick={() => setCurrentStep(2)}
                  className="px-6 py-3 rounded-lg bg-gray-700 hover:bg-gray-600 text-white transition-all duration-300 flex items-center gap-2"
                >
                  <FontAwesomeIcon icon={faArrowLeft} />
                  <span>Back</span>
                </button>

                <button
                  onClick={generateDAT}
                  className="px-8 py-3 rounded-lg bg-primary hover:bg-primary/90 text-white transition-all duration-300 flex items-center gap-2 shadow-lg shadow-primary/20"
                  disabled={loading || !uploadComplete}
                >
                  {loading ? (
                    <>
                      <FontAwesomeIcon icon={faSpinner} spin />
                      <span>Generating...</span>
                    </>
                  ) : (
                    <>
                      <FontAwesomeIcon icon={faCloudUploadAlt} />
                      <span>Generate DAT</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Step 4: Generate DAT */}
          {currentStep === 4 && (
            <div>
              <h2 className="text-2xl font-bold mb-6">Generating DAT</h2>
              <p className="text-gray-400 mb-8">
                Your privacy data is being processed to generate a DAT (Data Anchoring Token).
              </p>

              {!datGenerated ? (
                <div className="flex flex-col items-center justify-center py-8">
                  <div className="w-16 h-16 rounded-full border-4 border-primary/30 border-t-primary animate-spin mb-4">
                    <FontAwesomeIcon
                      icon={faLink}
                      className="text-2xl text-primary absolute inset-0 flex items-center justify-center"
                    />
                  </div>
                  <p className="text-gray-400">Processing your request...</p>
                </div>
              ) : (
                <div className="mb-8 p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-green-500/20 flex items-center justify-center">
                      <FontAwesomeIcon icon={faCheck} className="text-green-500" />
                    </div>
                    <div>
                      <h3 className="font-medium">DAT Generated Successfully!</h3>
                      <p className="text-sm text-gray-400">
                        Your DAT has been created and recorded on the blockchain.
                      </p>
                    </div>
                  </div>

                  <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 rounded-lg bg-gray-900 border border-gray-700">
                      <h3 className="text-sm font-medium text-gray-400 mb-1">File ID</h3>
                      <p className="font-medium">{fileId}</p>
                    </div>

                    <div className="p-4 rounded-lg bg-gray-900 border border-gray-700">
                      <h3 className="text-sm font-medium text-gray-400 mb-1">Job ID</h3>
                      <p className="font-medium">{jobId}</p>
                    </div>

                    <div className="p-4 rounded-lg bg-gray-900 border border-gray-700">
                      <h3 className="text-sm font-medium text-gray-400 mb-1">Node Address</h3>
                      <p className="text-sm font-medium">{shortenAddress(nodeAddress)}</p>
                    </div>

                    <div className="p-4 rounded-lg bg-gray-900 border border-gray-700">
                      <h3 className="text-sm font-medium text-gray-400 mb-1">Status</h3>
                      <p className="font-medium text-green-500">Completed</p>
                    </div>
                  </div>
                </div>
              )}

              <div className="mt-8 flex justify-center">
                <button
                  onClick={resetApp}
                  className="px-8 py-3 rounded-lg bg-primary hover:bg-primary/90 text-white transition-all duration-300 flex items-center gap-2 shadow-lg shadow-primary/20"
                  disabled={loading}
                >
                  <FontAwesomeIcon icon={faRefresh} />
                  <span>Start Over</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="py-4 px-6 border-t border-gray-800 text-center text-sm text-gray-500">
        <p>© 2025 LazAI. All rights reserved.</p>
      </footer>

      {}
      {notification.show && (
        <div
          className={`fixed top-4 right-4 max-w-sm w-full p-4 rounded-lg shadow-xl transform transition-all duration-300 ${
            notification.type === 'success'
              ? 'bg-green-500/10 border border-green-500/20 text-green-400'
              : 'bg-red-500/10 border border-red-500/20 text-red-400'
          }`}
        >
          <div className="flex">
            <div className="flex-shrink-0">
              <FontAwesomeIcon
                icon={notification.type === 'success' ? faCheck : faExclamationCircle}
                className={`h-5 w-5 ${notification.type === 'success' ? 'text-green-400' : 'text-red-400'}`}
              />
            </div>
            <div className="ml-3 flex-1">
              <p className="text-sm font-medium">{notification.type === 'success' ? 'Success' : 'Error'}</p>
              <p className="mt-1 text-sm">{notification.message}</p>
            </div>
            <div className="ml-4 flex-shrink-0 flex">
              <button
                onClick={() => setNotification({ show: false, type: 'success', message: '' })}
                className="text-gray-400 hover:text-white"
              >
                <FontAwesomeIcon icon={faTimes} className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
