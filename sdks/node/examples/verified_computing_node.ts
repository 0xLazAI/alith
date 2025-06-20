import axios, { type AxiosResponse } from "axios";
const proofRequest = {
  job_id: 16,
  file_id: 17,
  file_url: "1",
  encryption_key: "2",
  encryption_seed: "3",
  nonce: null,
  proof_url: null,
};
const response: AxiosResponse = await axios.post(
  "http://127.0.0.1:80/proof",
  proofRequest,
  {
    headers: { "Content-Type": "application/json" },
  }
);
if (response.status === 200) {
  console.log("Proof request sent successfully");
} else {
  console.log("Failed to send proof request:", response.data);
}
