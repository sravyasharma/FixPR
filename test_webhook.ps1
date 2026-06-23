$body = @{
    action = "opened"
    repository = @{
        id = 12345
        full_name = "sravyasharma/Vault"
        clone_url = "https://github.com/sravyasharma/Vault.git"
        default_branch = "main"
        language = "Python"
    }
    pull_request = @{
        number = 1
        title = "Testing AI Review"
        body = "Trigger review"
        html_url = "https://github.com/sravyasharma/Vault/pull/1"

        user = @{
            login = "sravyasharma"
        }

        head = @{
            sha = "main"
            ref = "main"
        }

        base = @{
            sha = "main"
            ref = "main"
        }
    }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/webhooks/github" `
    -Method POST `
    -ContentType "application/json" `
    -Headers @{
        "x-github-event"="pull_request"
        "x-github-delivery"="vault-test"
        "x-hub-signature-256"="test"
    } `
    -Body $body